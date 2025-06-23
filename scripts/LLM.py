#!/usr/bin/env python3

import os
import threading
import time
from typing import Optional, Callable
from queue import Queue, Empty
from openai import OpenAI


class LLMProcessor:
    """
    Language Model processor using OpenAI's API to generate responses from transcribed text.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 max_queue_size: int = 10,
                 system_prompt: Optional[str] = None):
        """
        Initialize the LLM processor.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
            max_queue_size: Maximum number of requests to queue
            system_prompt: System prompt to set context for the LLM
        """
        self.model = model
        self.max_queue_size = max_queue_size
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
        
        # Set system prompt
        self.system_prompt = system_prompt or (
            "You are a helpful assistant for a NAO robot. "
            "Respond to user inputs in a natural, friendly, and concise manner. "
            "Keep responses brief but informative."
        )
        
        # Processing queue and worker thread
        self.request_queue = Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.is_enabled = True
        self.is_running = False
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'requests_failed': 0,
            'total_tokens_used': 0,
            'requests_queued': 0
        }
        
        print(f"🧠 LLM Processor initialized with model: {self.model}")
    
    def start_worker(self):
        """
        Start the background worker thread for processing LLM requests.
        """
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("🚀 LLM worker thread started")
    
    def stop_worker(self):
        """
        Stop the background worker thread.
        """
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        print("🛑 LLM worker thread stopped")
    
    def add_request(self, text: str, callback: Optional[Callable[[str], None]] = None):
        """
        Add a text for LLM processing.
        
        Args:
            text: Input text to process
            callback: Optional callback function to call with the response
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
                'timestamp': time.time()
            }
            self.request_queue.put(request, block=False)
            self.stats['requests_queued'] += 1
            print(f"📝 Added text to LLM queue: '{text[:50]}{'...' if len(text) > 50 else ''}'")
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
        Process a single LLM request.
        
        Args:
            request: Request dictionary containing text and callback
        """
        try:
            text = request['text']
            callback = request['callback']
            
            print(f"🤖 Processing LLM request: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Create the chat completion
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=150,  # Keep responses concise
                temperature=0.7
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Update statistics
            self.stats['requests_processed'] += 1
            if hasattr(response, 'usage') and response.usage:
                self.stats['total_tokens_used'] += response.usage.total_tokens
            
            print(f"✅ LLM Response: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
            
            # Call callback if provided
            if callback:
                callback(response_text)
            
        except Exception as e:
            self.stats['requests_failed'] += 1
            error_msg = f"❌ LLM processing failed: {str(e)}"
            print(error_msg)
            
            # Call callback with error if provided
            if callback:
                callback(f"Error: {str(e)}")
    
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
        return {
            'enabled': self.is_enabled,
            'model': self.model,
            'queue_size': self.get_queue_size(),
            'requests_processed': self.stats['requests_processed'],
            'requests_failed': self.stats['requests_failed'],
            'total_tokens_used': self.stats['total_tokens_used'],
            'requests_queued': self.stats['requests_queued']
        }


if __name__ == '__main__':
    # Test the LLM processor
    llm = LLMProcessor()
    
    def print_response(response: str):
        print(f"🗣️  LLM Response: {response}")
    
    llm.start_worker()
    
    # Test request
    llm.add_request("Tell me a three sentence bedtime story about a unicorn.", print_response)
    
    # Wait for completion
    llm.wait_for_completion()
    llm.stop_worker()
    
    print(f"📊 Final stats: {llm.get_stats()}") 