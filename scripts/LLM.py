#!/usr/bin/env python3

import os
import threading
import time
import json
import pickle
from typing import Optional, Callable, Dict, List, TypedDict, Any
from queue import Queue, Empty
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel, Field
from prompt import AnimationPrompt, AnimationActions, RespondAgentPrompt, concept_dict
from animation_executor import NAOAnimationExecutor
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
LANGGRAPH_AVAILABLE = True



class ConversationState(TypedDict, total=False):
    """State schema for conversation tracking"""
    messages: List[dict]
    user_id: str
    session_id: str
    context: Dict[str, Any]
    last_activity: float
    _temp_animation_actions: List[str]  # Temporary field for animation actions
    _skip_tts: bool  # Temporary field to indicate if TTS should be skipped


class PersistentLLMProcessor:
    """
    Enhanced LLM processor with LangGraph persistence for conversation state management.
    Automatically saves conversation state after 10 seconds of inactivity.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 respond_model: Optional[str] = None,
                 animation_model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_queue_size: int = 10,
                 system_prompt: Optional[str] = None,
                 max_conversation_length: int = 20,
                 save_directory: str = "/tmp/nao_conversations",
                 auto_save_delay: float = 10.0,
                 nao_ip: str = os.getenv("NAO_IP", "127.0.0.1")):
        """
        Initialize the LLM processor with LangGraph persistence.
        
        Args:
            model: Default OpenAI model to use (fallback for both agents)
            respond_model: Specific model for the respond agent (if None, uses model)
            animation_model: Specific model for the animation agent (if None, uses model)
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
            max_queue_size: Maximum number of requests to queue
            system_prompt: System prompt to set context for the LLM
            max_conversation_length: Maximum number of messages to keep in conversation history
            save_directory: Directory to save conversation states
            auto_save_delay: Seconds of inactivity before auto-saving (default: 10.0)
            nao_ip: IP address or hostname of the NAO robot for animation execution
        """
        self.model = model
        self.respond_model = respond_model or model
        self.animation_model = animation_model or model
        self.max_queue_size = max_queue_size
        self.max_conversation_length = max_conversation_length
        self.save_directory = Path(save_directory)
        self.auto_save_delay = auto_save_delay
        
        # Create save directory if it doesn't exist
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client (kept for backward compatibility)
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
        
        # Initialize the two agents using LangChain's init_chat_model with different models
        if LANGGRAPH_AVAILABLE:
            try:
                # RespondAgent - for generating conversational responses
                self.respond_agent = init_chat_model(self.respond_model, model_provider="openai")
                
                # AnimationAgent - for analyzing responses and selecting animations
                self.animation_agent = init_chat_model(self.animation_model, model_provider="openai")
                
                if self.respond_model == self.animation_model:
                    print(f"✅ Initialized RespondAgent and AnimationAgent with model: {self.respond_model}")
                else:
                    print(f"✅ Initialized RespondAgent with model: {self.respond_model}")
                    print(f"✅ Initialized AnimationAgent with model: {self.animation_model}")
            except Exception as e:
                print(f"⚠️  Failed to initialize LangChain agents, falling back to OpenAI client: {e}")
                self.respond_agent = None
                self.animation_agent = None
        else:
            self.respond_agent = None
            self.animation_agent = None
        
        # Initialize NAO animation executor
        try:
            self.animation_executor = NAOAnimationExecutor(nao_ip=nao_ip)
            print(f"🤖 NAO Animation Executor initialized for IP: {nao_ip}")
        except Exception as e:
            print(f"⚠️  Failed to initialize NAO Animation Executor: {e}")
            self.animation_executor = None
        
        self.system_prompt = RespondAgentPrompt
        
        # Initialize LangGraph persistence components
        self.checkpointer = MemorySaver()
        self.memory_store = InMemoryStore()
        
        # Initialize LangGraph
        self.conversation_graphs = {}  # Dict[str, CompiledStateGraph]
        self.conversation_configs = {}  # Dict[str, dict] - LangGraph configs for each session
        self.last_activity = {}  # Dict[str, float] - Track last activity per session
        self.pending_save = {}  # Dict[str, bool] - Track if session needs saving
        self.auto_save_scheduled = {}  # Dict[str, bool] - Track if auto-save is scheduled for session
        
        # Processing queue and worker thread
        self.request_queue = Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.auto_save_thread = None
        self.is_enabled = True
        self.is_running = False
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'requests_failed': 0,
            'total_tokens_used': 0,
            'requests_queued': 0,
            'active_conversations': 0,
            'conversations_saved': 0,
            'conversations_loaded': 0
        }
        
        if self.respond_model == self.animation_model:
            print(f"🧠 Persistent LLM Processor initialized with model: {self.respond_model}")
        else:
            print(f"🧠 Persistent LLM Processor initialized")
            print(f"   🗣️  RespondAgent model: {self.respond_model}")
            print(f"   🎭 AnimationAgent model: {self.animation_model}")
        print(f"💾 Save directory: {self.save_directory}")
        print(f"⏰ Auto-save delay: {self.auto_save_delay}s")
    
    def _create_conversation_graph(self, session_id: str):
        """
        Create a new conversation graph for a session with persistence.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Compiled state graph for the conversation
        """
        def respond_agent_node(state: ConversationState) -> ConversationState:
            """RespondAgent: Process a message and generate response using LangChain agent"""
            messages = state["messages"]
            
            try:
                # Use LangChain agent if available, otherwise fall back to OpenAI client
                if self.respond_agent:
                    # Build conversation history for LangChain
                    langchain_messages = [SystemMessage(content=self.system_prompt)]
                    
                    # Add conversation history (limit to max_conversation_length)
                    for msg in messages[-(self.max_conversation_length-1):]:  # -1 for system message
                        if msg["role"] == "user":
                            langchain_messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            langchain_messages.append(AIMessage(content=msg["content"]))
                    
                    # Generate response using LangChain agent
                    response = self.respond_agent.invoke(langchain_messages)
                    response_text = response.content.strip()
                    
                else:
                    # Fallback to OpenAI client
                    openai_messages = [{"role": "system", "content": self.system_prompt}]
                    
                    # Add conversation history (limit to max_conversation_length)
                    for msg in messages[-(self.max_conversation_length-1):]:  # -1 for system message
                        openai_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Generate response
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=openai_messages,
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    response_text = response.choices[0].message.content.strip()
                    
                    # Update statistics for OpenAI usage
                    if hasattr(response, 'usage') and response.usage:
                        self.stats['total_tokens_used'] += response.usage.total_tokens
                
                # Add assistant response to conversation
                messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "timestamp": time.time()
                })
                
                # Update statistics
                self.stats['requests_processed'] += 1
                
                # Update last activity and mark as needing save
                current_time = time.time()
                state["last_activity"] = current_time
                self.last_activity[session_id] = current_time
                self.pending_save[session_id] = True  # Mark this session as needing save
                
                # Schedule auto-save if not already scheduled
                if not self.auto_save_scheduled.get(session_id, False):
                    self.auto_save_scheduled[session_id] = True
                
                return {
                    **state,
                    "messages": messages,
                    "last_activity": current_time
                }
                
            except Exception as e:
                self.stats['requests_failed'] += 1
                error_response = f"I'm sorry, I encountered an error: {str(e)}"
                
                messages.append({
                    "role": "assistant", 
                    "content": error_response,
                    "timestamp": time.time(),
                    "error": True
                })
                
                current_time = time.time()
                state["last_activity"] = current_time
                self.last_activity[session_id] = current_time
                self.pending_save[session_id] = True  # Mark this session as needing save
                
                # Schedule auto-save if not already scheduled
                if not self.auto_save_scheduled.get(session_id, False):
                    self.auto_save_scheduled[session_id] = True
                
                return {
                    **state,
                    "messages": messages,
                    "last_activity": current_time
                }
        
        def animation_agent_node(state: ConversationState) -> ConversationState:
            """AnimationAgent: Analyze the assistant's response and select appropriate animation actions using LangChain agent"""
            messages = state["messages"]
            animation_actions = []
            
            # Get the latest assistant message
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            if assistant_messages:
                latest_response = assistant_messages[-1]["content"]
                
                try:
                    # Check if the response is a direct action command
                    available_action_keys = list(concept_dict.keys())
                    if latest_response.strip() in available_action_keys:
                        # This is a direct action command, execute the specific animation
                        action_key = latest_response.strip()
                        print(f"🎯 Direct action command detected: {action_key}")
                        
                        # Get random animation from the concept_dict for this action
                        import random
                        action_animations = concept_dict[action_key]
                        selected_animation = random.choice(action_animations)
                        
                        # Execute the animation directly using animation executor
                        if self.animation_executor:
                            # Execute the selected animation path directly
                            success = self.animation_executor.execute_direct_animation(action_key, selected_animation)
                            if success:
                                print(f"🎭 Successfully executed direct action: {action_key} -> {selected_animation}")
                            else:
                                print(f"❌ Failed to execute direct action: {action_key} -> {selected_animation}")
                        
                        # Don't generate additional animations for direct commands
                        animation_actions = []
                        
                        # Mark this as a direct action so TTS can be skipped
                        state["_skip_tts"] = True
                        print(f"🔇 Skipping TTS for direct action: {action_key}")
                        
                    else:
                        # Normal conversation response - analyze for appropriate animations
                        if self.animation_agent:
                            # Use LangChain agent with structured output for animation analysis
                            animation_prompt_text = AnimationPrompt.format(response_text=latest_response)
                            
                            # Create structured LLM that returns AnimationActions
                            structured_animation_agent = self.animation_agent.with_structured_output(AnimationActions)
                            
                            # Get structured response
                            animation_response = structured_animation_agent.invoke([HumanMessage(content=animation_prompt_text)])
                            
                            # Extract actions from structured response
                            animation_actions = animation_response.actions
                            
                            # Validate actions against available set
                            valid_actions = ["affirmative_context", "anterior", "comparison", "confirmation", "disappointment", "diversity", "exclamation", "joy", "people", "self", "user"]
                            animation_actions = [action for action in animation_actions if action in valid_actions]
                        
                        else:
                            # Fallback when animation agent is not available
                            print(f"⚠️  Animation agent not available, using default action")
                            animation_actions = ["confirmation"]  # Default fallback
                    
                    print(f"🎭 Animation Analysis for session {session_id}:")
                    print(f"   Response: '{latest_response[:100]}{'...' if len(latest_response) > 100 else ''}'")
                    print(f"   Selected actions: {animation_actions}")
                    
                except Exception as e:
                    print(f"⚠️  Animation analysis failed: {e}")
                    animation_actions = ["confirmation"]  # Default fallback
            
            # Store animation actions temporarily for the callback (not in persistent state)
            state["_temp_animation_actions"] = animation_actions
            
            # Ensure _skip_tts is set if not already (default to False for normal responses)
            if "_skip_tts" not in state:
                state["_skip_tts"] = False
            
            return state
        
        # Create the graph with both agents
        workflow = StateGraph(ConversationState)
        workflow.add_node("respond_agent", respond_agent_node)
        workflow.add_node("animation_agent", animation_agent_node)
        workflow.set_entry_point("respond_agent")
        workflow.add_edge("respond_agent", "animation_agent")
        workflow.add_edge("animation_agent", END)
        
        # Compile with checkpointer and memory store
        return workflow.compile(
            checkpointer=self.checkpointer,
            store=self.memory_store
        )
    
    def _get_or_create_conversation(self, session_id: str = "default", user_id: str = "default"):
        """
        Get existing conversation or create a new one, with persistence support.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Conversation graph for the session
        """
        if session_id not in self.conversation_graphs:
            # Try to load existing conversation state first
            loaded_state = self.load_conversation_state(session_id)
            
            self.conversation_graphs[session_id] = self._create_conversation_graph(session_id)
            
            # Create config for this session
            self.conversation_configs[session_id] = {
                "configurable": {
                    "thread_id": session_id,
                    "user_id": user_id
                }
            }
            
            if loaded_state:
                print(f"📂 Loaded existing conversation for session: {session_id}")
                # Restore the state in the graph
                try:
                    graph = self.conversation_graphs[session_id]
                    config = self.conversation_configs[session_id]
                    
                    # Update the graph state with loaded data
                    graph.update_state(config, loaded_state)
                    
                    self.stats['conversations_loaded'] += 1
                except Exception as e:
                    print(f"⚠️  Failed to restore conversation state: {e}")
                    # Continue with new conversation if restoration fails
            else:
                print(f"🆕 Created new conversation for session: {session_id}")
            
            self.stats['active_conversations'] += 1
            current_time = time.time()
            self.last_activity[session_id] = current_time
            
            # Initialize auto-save flags for new conversations
            self.pending_save[session_id] = False
            self.auto_save_scheduled[session_id] = False
        
        return self.conversation_graphs[session_id]
    
    def save_conversation_state(self, session_id: str) -> bool:
        """
        Save conversation state to disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if saved successfully
        """
        try:
            if session_id not in self.conversation_graphs:
                return False
            
            graph = self.conversation_graphs[session_id]
            config = self.conversation_configs[session_id]
            
            # Get current state from the graph
            state = graph.get_state(config)
            
            if state and hasattr(state, 'values'):
                # Prepare state data for saving
                state_data = {
                    'session_id': session_id,
                    'user_id': config["configurable"]["user_id"],
                    'messages': state.values.get("messages", []),
                    'context': state.values.get("context", {}),
                    'last_activity': state.values.get("last_activity", time.time()),
                    'saved_at': time.time(),
                    'model': self.model,
                    'system_prompt': self.system_prompt
                }
                
                # Save to file
                save_path = self.save_directory / f"{session_id}.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                
                self.stats['conversations_saved'] += 1
                
                # Clear auto-save flags when manually saved
                self.pending_save[session_id] = False
                self.auto_save_scheduled[session_id] = False
                
                print(f"💾 Saved conversation state for session: {session_id}")
                return True
            
        except Exception as e:
            print(f"❌ Failed to save conversation state for {session_id}: {e}")
        
        return False
    
    def load_conversation_state(self, session_id: str) -> Optional[dict]:
        """
        Load conversation state from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Loaded state data or None if not found
        """
        try:
            load_path = self.save_directory / f"{session_id}.json"
            
            if not load_path.exists():
                return None
            
            with open(load_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Validate loaded data
            if (state_data.get('session_id') == session_id and
                'messages' in state_data):
                
                print(f"📂 Loaded conversation state for session: {session_id}")
                print(f"   Messages: {len(state_data['messages'])}")
                print(f"   Last activity: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state_data.get('last_activity', 0)))}")
                
                return {
                    'messages': state_data['messages'],
                    'user_id': state_data.get('user_id', 'default'),
                    'session_id': session_id,
                    'context': state_data.get('context', {}),
                    'last_activity': state_data.get('last_activity', time.time())
                }
            
        except Exception as e:
            print(f"❌ Failed to load conversation state for {session_id}: {e}")
        
        return None
    
    def list_saved_conversations(self) -> List[dict]:
        """
        List all saved conversation sessions.
        
        Returns:
            List of conversation metadata
        """
        conversations = []
        
        try:
            for file_path in self.save_directory.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        state_data = json.load(f)
                    
                    conversations.append({
                        'session_id': state_data.get('session_id'),
                        'user_id': state_data.get('user_id'),
                        'message_count': len(state_data.get('messages', [])),
                        'last_activity': state_data.get('last_activity'),
                        'saved_at': state_data.get('saved_at'),
                        'model': state_data.get('model'),
                        'file_path': str(file_path)
                    })
                except Exception as e:
                    print(f"⚠️  Failed to read conversation file {file_path}: {e}")
                    
        except Exception as e:
            print(f"❌ Failed to list conversations: {e}")
        
        return sorted(conversations, key=lambda x: x.get('last_activity', 0), reverse=True)
    
    def delete_conversation_state(self, session_id: str) -> bool:
        """
        Delete saved conversation state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            file_path = self.save_directory / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️  Deleted conversation state for session: {session_id}")
                return True
        except Exception as e:
            print(f"❌ Failed to delete conversation state for {session_id}: {e}")
        
        return False
    
    def _auto_save_loop(self):
        """
        Smart auto-save loop that only saves when there's been conversation activity
        followed by inactivity for the specified delay period.
        """
        print(f"💾 Smart auto-save loop started (delay: {self.auto_save_delay}s)")
        
        while self.is_running:
            try:
                current_time = time.time()
                sessions_to_save = []
                
                # Check sessions that need saving and have been inactive long enough
                for session_id in list(self.pending_save.keys()):
                    # Only check sessions that are marked as needing save and are scheduled
                    if (self.pending_save.get(session_id, False) and 
                        self.auto_save_scheduled.get(session_id, False)):
                        
                        last_activity = self.last_activity.get(session_id, 0)
                        
                        # Check if enough time has passed since last activity
                        if (current_time - last_activity) >= self.auto_save_delay:
                            sessions_to_save.append(session_id)
                
                # Save sessions that meet the criteria
                for session_id in sessions_to_save:
                    if session_id in self.conversation_graphs:
                        success = self.save_conversation_state(session_id)
                        if success:
                            # Clear the pending save flags after successful save
                            self.pending_save[session_id] = False
                            self.auto_save_scheduled[session_id] = False
                            print(f"✅ Auto-saved conversation for session: {session_id}")
                        else:
                            print(f"❌ Auto-save failed for session: {session_id}")
                
                # Sleep for a short interval (check every 2 seconds)
                time.sleep(2.0)
                
            except Exception as e:
                print(f"❌ Error in auto-save loop: {e}")
                time.sleep(5.0)
        
        print("💾 Smart auto-save loop stopped")
    
    def start_worker(self):
        """
        Start the background worker thread for processing LLM requests and auto-saving.
        """
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.is_running = True
        
        # Start main worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Start auto-save thread
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()
        
        # Start animation executor worker if available
        if self.animation_executor:
            self.animation_executor.start_worker()
        
        print("🚀 Persistent LLM worker threads started with smart auto-save and animation execution")
    
    def stop_worker(self):
        """
        Stop the background worker threads and save all active conversations.
        """
        self.is_running = False
        
        # Save all active conversations before stopping
        print("💾 Saving all active conversations...")
        for session_id in list(self.conversation_graphs.keys()):
            self.save_conversation_state(session_id)
        
        # Stop animation executor if available
        if self.animation_executor:
            self.animation_executor.stop_worker()
        
        # Wait for threads to stop
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        if self.auto_save_thread and self.auto_save_thread.is_alive():
            self.auto_save_thread.join(timeout=5.0)
            
        print("🛑 Persistent LLM worker threads and animation executor stopped")
    
    def add_request(self, text: str, callback: Optional[Callable[[str], None]] = None, 
                   session_id: str = "default", user_id: str = "default"):
        """
        Add a text for LLM processing with conversation state.
        
        Args:
            text: Input text to process
            callback: Optional callback function to call with the response
            session_id: Session identifier for conversation tracking
            user_id: User identifier
        """
        if not self.is_enabled:
            print("⚠️  LLM processing is disabled")
            return
        
        if not text.strip():
            print("⚠️  Empty text provided for LLM processing")
            return
        
        try:
            request = {
                'text': text.strip(),
                'callback': callback,
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': time.time()
            }
            self.request_queue.put(request, block=False)
            self.stats['requests_queued'] += 1
            
            # Update last activity and mark for auto-save
            current_time = time.time()
            self.last_activity[session_id] = current_time
            self.pending_save[session_id] = True  # Mark this session as needing save
            
            # Schedule auto-save if not already scheduled
            if not self.auto_save_scheduled.get(session_id, False):
                self.auto_save_scheduled[session_id] = True
            
            print(f"📝 Added text to LLM queue (session: {session_id}): '{text[:50]}{'...' if len(text) > 50 else ''}'")
        except:
            print(f"⚠️  LLM queue is full, dropping request")
    
    def _worker_loop(self):
        """
        Background worker loop for processing LLM requests.
        """
        print("🔄 LLM worker loop started")
        
        while self.is_running:
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=1.0)
                self._process_request(request)
                self.request_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Error in LLM worker loop: {str(e)}")
        
        print("🔄 LLM worker loop stopped")
    
    def _process_request(self, request: dict):
        """
        Process a single LLM request using LangGraph with persistence.
        
        Args:
            request: Request dictionary containing text, callback, and session info
        """
        try:
            text = request['text']
            callback = request['callback']
            session_id = request['session_id']
            user_id = request['user_id']
            
            print(f"🤖 Processing LLM request (session: {session_id}): '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Get or create conversation graph
            graph = self._get_or_create_conversation(session_id, user_id)
            config = self.conversation_configs[session_id]
            
            # Prepare input state
            input_state = {
                "messages": [{
                    "role": "user", 
                    "content": text,
                    "timestamp": time.time()
                }],
                "user_id": user_id,
                "session_id": session_id,
                "context": {},
                "last_activity": time.time()
            }
            
            # Get current state and append new message
            try:
                current_state = graph.get_state(config)
                if current_state and hasattr(current_state, 'values') and current_state.values.get("messages"):
                    existing_messages = current_state.values["messages"]
                    input_state["messages"] = existing_messages + input_state["messages"]
                    input_state["context"] = current_state.values.get("context", {})
            except Exception as e:
                print(f"⚠️  Could not retrieve existing state: {e}")
            
            # Process through LangGraph
            result = graph.invoke(input_state, config)
            
            # Get the latest assistant response and animation actions
            assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
            animation_actions = result.get("_temp_animation_actions", [])
            
            if assistant_messages:
                response_text = assistant_messages[-1]["content"]
                print(f"✅ LLM Response (session: {session_id}): '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
                print(f"🎭 Animation Actions: {animation_actions}")
                
                # Update last activity (response completed, ready for auto-save scheduling)
                current_time = time.time()
                self.last_activity[session_id] = current_time
                
                # Execute animations on NAO robot if executor is available
                if self.animation_executor and animation_actions:
                    self.animation_executor.add_animation_request(
                        actions=animation_actions,
                        delay_between=0.7 # 0.7 second delay between animations
                    )
                
                # Call callback if provided - now includes both response and actions
                if callback:
                    # Get skip_tts flag from the result
                    skip_tts = result.get("_skip_tts", False)
                    
                    # Create a combined response with animation info
                    enhanced_response = {
                        "text": response_text,
                        "animation_actions": animation_actions,
                        "session_id": session_id,
                        "skip_tts": skip_tts
                    }
                    callback(enhanced_response)
            else:
                print(f"⚠️  No assistant response generated")
                if callback:
                    callback({
                        "text": "I'm sorry, I couldn't generate a response.",
                        "animation_actions": [],
                        "session_id": session_id,
                        "skip_tts": False
                    })
            
        except Exception as e:
            self.stats['requests_failed'] += 1
            error_msg = f"❌ LLM processing failed: {str(e)}"
            print(error_msg)
            
            # Call callback with error if provided
            if 'callback' in request and request['callback']:
                request['callback']({
                    "text": f"Error: {str(e)}",
                    "animation_actions": ["sad"],
                    "session_id": request.get('session_id', 'default'),
                    "skip_tts": False
                })
    
    def get_conversation_history(self, session_id: str = "default") -> List[dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages in the conversation
        """
        try:
            if session_id in self.conversation_graphs:
                config = self.conversation_configs[session_id]
                state = self.conversation_graphs[session_id].get_state(config)
                if state and hasattr(state, 'values'):
                    return state.values.get("messages", [])
            
            # Try to load from disk if not in memory
            loaded_state = self.load_conversation_state(session_id)
            if loaded_state:
                return loaded_state.get("messages", [])
                
        except Exception as e:
            print(f"⚠️  Error getting conversation history: {e}")
        
        return []
    

    
    def clear_conversation(self, session_id: str = "default"):
        """
        Clear conversation history for a session and delete saved state.
        
        Args:
            session_id: Session identifier
        """
        try:
            if session_id in self.conversation_graphs:
                config = self.conversation_configs[session_id]
                # Clear the state in the graph
                empty_state = {
                    "messages": [],
                    "user_id": config["configurable"]["user_id"],
                    "session_id": session_id,
                    "context": {},
                    "last_activity": time.time()
                }
                self.conversation_graphs[session_id].update_state(config, empty_state)
            
            # Delete saved file
            self.delete_conversation_state(session_id)
            
            # Reset auto-save flags
            self.pending_save[session_id] = False
            self.auto_save_scheduled[session_id] = False
            
            print(f"🗑️  Cleared conversation history for session: {session_id}")
        except Exception as e:
            print(f"⚠️  Error clearing conversation: {e}")
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session identifiers
        """
        return list(self.conversation_graphs.keys())
    
    def get_queue_size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of requests in queue
        """
        return self.request_queue.qsize()
    
    def is_queue_empty(self) -> bool:
        """
        Check if the processing queue is empty.
        
        Returns:
            True if queue is empty
        """
        return self.request_queue.empty()
    
    def wait_for_completion(self, timeout: float = 30.0):
        """
        Wait for all queued requests to be processed.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        while not self.is_queue_empty() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.is_queue_empty():
            print(f"⏰ Timeout waiting for LLM queue completion")
        else:
            print(f"✅ All LLM requests processed")
    
    def enable(self):
        """Enable LLM processing."""
        self.is_enabled = True
        if not self.is_running:
            self.start_worker()
        print("✅ LLM processing enabled")
    
    def disable(self):
        """Disable LLM processing."""
        self.is_enabled = False
        print("⏸️  LLM processing disabled")
    
    def get_stats(self) -> dict:
        """
        Get LLM processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'enabled': self.is_enabled,
            'model': self.model,
            'respond_model': self.respond_model,
            'animation_model': self.animation_model,
            'queue_size': self.get_queue_size(),
            'requests_processed': self.stats['requests_processed'],
            'requests_failed': self.stats['requests_failed'],
            'total_tokens_used': self.stats['total_tokens_used'],
            'requests_queued': self.stats['requests_queued'],
            'active_conversations': self.stats['active_conversations'],
            'active_sessions': len(self.conversation_graphs),
            'conversations_saved': self.stats['conversations_saved'],
            'conversations_loaded': self.stats['conversations_loaded'],
            'save_directory': str(self.save_directory),
            'auto_save_delay': self.auto_save_delay
        }
        
        # Add animation executor stats if available
        if self.animation_executor:
            stats.update({
                'animation_executor_enabled': True,
                'animation_queue_size': self.animation_executor.get_queue_size(),
                'animation_executor_busy': self.animation_executor.is_busy(),
                'available_animations': self.animation_executor.get_available_actions()
            })
        else:
            stats['animation_executor_enabled'] = False
            
        return stats
    
    def update_nao_ip(self, nao_ip: str):
        """
        Update NAO robot IP address for animation execution.
        
        Args:
            nao_ip: New IP address or hostname
        """
        if self.animation_executor:
            self.animation_executor.update_nao_ip(nao_ip)
        else:
            print("⚠️  Animation executor not available")
    
    def test_nao_connection(self) -> bool:
        """
        Test connection to NAO robot.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.animation_executor:
            return self.animation_executor.test_connection()
        else:
            print("⚠️  Animation executor not available")
            return False


# Alias for backward compatibility
LLMProcessor = PersistentLLMProcessor


if __name__ == '__main__':
    # Test the persistent LLM processor
    llm = PersistentLLMProcessor(save_directory="/tmp/test_conversations")
    
    def print_response(response):
        if isinstance(response, dict):
            print(f"🗣️  LLM Response: {response['text']}")
            print(f"🎭 Animation Actions: {response['animation_actions']}")
        else:
            print(f"🗣️  LLM Response: {response}")
    
    llm.start_worker()
    
    # Test conversation with persistence
    session_id = "test_session_persistence"
    llm.add_request("Hello, my name is Alice.", print_response, session_id)
    time.sleep(2)  # Wait for processing
    llm.add_request("What's my name?", print_response, session_id)  # Should remember Alice
    time.sleep(2)
    
    # Wait for completion
    llm.wait_for_completion()
    
    # Print conversation history
    history = llm.get_conversation_history(session_id)
    print(f"💬 Conversation History: {len(history)} messages")
    for msg in history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Test auto-save (wait for 11 seconds to trigger auto-save)
    print(f"⏰ Waiting {llm.auto_save_delay + 1} seconds for auto-save...")
    time.sleep(llm.auto_save_delay + 1)
    
    # List saved conversations
    saved_conversations = llm.list_saved_conversations()
    print(f"💾 Saved conversations: {len(saved_conversations)}")
    for conv in saved_conversations:
        print(f"  Session: {conv['session_id']}, Messages: {conv['message_count']}")
    
    llm.stop_worker()
    
    print(f"📊 Final stats: {llm.get_stats()}") 