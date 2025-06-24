#!/usr/bin/env python3

import rclpy
import signal
import sys
import time
from typing import Optional
from Audio import NaoAudioRecorder, SAVE_DIRECTORY
from STT import STTTranscriber
from LLM import PersistentLLMProcessor
from TTS import TTSProcessor
from AudioPlayer import NaoAudioPlayer


class NaoAudioSTTSystem:
    """
    Main system that coordinates touch-triggered audio recording, speech-to-text transcription, 
    LLM processing with conversation state management, and text-to-speech synthesis.
    """
    
    def __init__(self):
        """
        Initialize the touch-triggered audio recording, STT, LLM, and TTS system.
        """
        print("🚀 Initializing NAO Touch-Triggered Audio + STT + LLM + TTS System")
        
        # Initialize audio player first
        self.audio_player = NaoAudioPlayer()
        
        # Initialize TTS processor
        self.tts = TTSProcessor(
            model="gpt-4o-mini-tts",
            voice="nova",
            output_dir="/tmp/nao_tts",
            max_queue_size=10
        )
        
        # Initialize LLM processor with conversation state management and persistence
        self.llm = PersistentLLMProcessor(
            model="gpt-4o-mini",
            max_queue_size=10,
            max_conversation_length=20,  # Keep last 20 messages for context
            system_prompt="You are a helpful assistant for a NAO robot. Respond to user speech in a natural, friendly, and concise manner. Keep responses brief but informative. Remember our conversation context and refer to previous messages when relevant.",
            save_directory="/tmp/nao_conversations",  # Directory for persistent conversation storage
            auto_save_delay=10.0  # Auto-save after 10 seconds of inactivity
        )
        
        # Initialize STT transcriber with LLM callback
        self.stt = STTTranscriber(
            model="gpt-4o-mini-transcribe", 
            max_queue_size=20,
            on_transcription_complete=self._on_transcription_complete
        )
        
        # Initialize touch-triggered audio recorder with callback to STT
        self.audio_recorder = NaoAudioRecorder(
            node_name='nao_audio_recorder',
            on_segment_saved=self._on_audio_segment_saved,
            enable_stt=True
        )
        
        # Conversation management
        self.current_session_id = "nao_session"  # Single session for NAO robot
        self.user_id = "nao_user"  # Default user for the robot
        
        # System state
        self.is_running = False
        
        print("✅ NAO Touch-Triggered Audio + STT + LLM + TTS System with Conversation State initialized")
    
    def _on_audio_segment_saved(self, file_path: str):
        """
        Callback function called when audio recorder saves a segment.
        Adds the file to STT transcription queue.
        
        Args:
            file_path: Path to the saved audio file
        """
        if self.stt.is_enabled:
            self.stt.add_file_for_transcription(file_path)
        else:
            print(f"📁 Audio saved: {file_path} (STT disabled)")
    
    def _on_transcription_complete(self, transcription: str):
        """
        Callback function called when STT transcription is complete.
        Sends the transcription to LLM for processing with conversation context.
        
        Args:
            transcription: The transcribed text
        """
        if self.llm.is_enabled and transcription.strip():
            # Use session and user IDs for conversation state management
            self.llm.add_request(
                transcription, 
                self._on_llm_response, 
                session_id=self.current_session_id,
                user_id=self.user_id
            )
    
    def _on_llm_response(self, response):
        """
        Callback function called when LLM processing is complete.
        Prints the LLM response and converts it to speech.
        
        Args:
            response: The LLM response (can be string or dict with text and animation_actions)
        """
        print(f"\n🤖 LLM RESPONSE: {response}")
        print("═" * 60)
        print()
        
        # Handle both string and dictionary responses
        if isinstance(response, dict):
            response_text = response.get('text', '')
            animation_actions = response.get('animation_actions', [])
            if animation_actions:
                print(f"🎭 Animation Actions: {animation_actions}")
        else:
            response_text = str(response)
        
        # Convert LLM response to speech and play on NAO
        if self.tts.is_enabled and response_text.strip() and not response_text.startswith("Error:"):
            self.tts.add_request(response_text, self._on_tts_complete, save_file=True)
    
    def _on_tts_complete(self, audio_file_path: str):
        """
        Callback function called when TTS processing is complete.
        Plays the generated audio file using the audio player.
        
        Args:
            audio_file_path: Path to the generated audio file
        """
        if audio_file_path and not audio_file_path.startswith("Error:"):
            print(f"🎵 TTS audio generated: {audio_file_path}")
            # Play the audio file using the audio player
            success = self.audio_player.play_audio_file(audio_file_path)
            if success:
                print(f"✅ Audio played successfully")
            else:
                print(f"❌ Audio playback failed")
        else:
            print(f"❌ TTS generation failed: {audio_file_path}")
    
    def start(self):
        """
        Start the touch-triggered audio recording, STT, LLM, and TTS system.
        """
        if self.is_running:
            print("⚠️  System is already running")
            return
        
        self.is_running = True
        
        # Start worker threads
        if self.llm.is_enabled:
            self.llm.start_worker()
        if self.tts.is_enabled:
            self.tts.start_worker()
        
        print(f"\n🎤 Starting NAO Touch-Triggered Audio Recording + STT + LLM + TTS System")
        print(f"🤚 Touch any head button to start recording, release to hear AI response")
        print(f"🧠 Conversation context is maintained across interactions")
        print(f"💾 Smart auto-save: saves once after {self.llm.auto_save_delay}s of conversation inactivity")
        print(f"📁 Audio recordings saved to: {SAVE_DIRECTORY}")
        print(f"💾 Conversation states saved to: {self.llm.save_directory}")
        print(f"🤖 STT Status: {'Enabled' if self.stt.is_enabled else 'Disabled'}")
        print(f"🧠 STT Model: {self.stt.model}")
        print(f"📋 STT Queue Size: {self.stt.get_queue_size()}")
        print(f"🤖 LLM Status: {'Enabled' if self.llm.is_enabled else 'Disabled'}")
        print(f"🧠 LLM Model: {self.llm.model}")
        print(f"💬 LLM Session ID: {self.current_session_id}")
        print(f"👤 LLM User ID: {self.user_id}")
        print(f"📋 LLM Queue Size: {self.llm.get_queue_size()}")
        print(f"🔊 TTS Status: {'Enabled' if self.tts.is_enabled else 'Disabled'}")
        print(f"🎵 TTS Model: {self.tts.model}")
        print(f"🗣️  TTS Voice: {self.tts.voice}")
        print(f"📋 TTS Queue Size: {self.tts.get_queue_size()}")
        print(f"⏹️  Press Ctrl+C to stop\n")
        
        try:
            # Start ROS2 spinning
            rclpy.spin(self.audio_recorder)
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop the touch-triggered audio recording, STT, LLM, and TTS system gracefully.
        """
        if not self.is_running:
            return
        
        print("🛑 Shutting down NAO Touch-Triggered Audio + STT + LLM + TTS System...")
        
        # Save any remaining audio buffer
        self.audio_recorder.save_remaining_buffer()
        
        # Wait for remaining transcriptions to complete
        if self.stt.is_enabled and not self.stt.is_queue_empty():
            print("⏳ Waiting for remaining transcriptions to complete...")
            try:
                self.stt.wait_for_completion()
            except KeyboardInterrupt:
                print("⚡ Force stopping transcriptions...")
        
        # Wait for remaining LLM requests to complete
        if self.llm.is_enabled and not self.llm.is_queue_empty():
            print("⏳ Waiting for remaining LLM requests to complete...")
            try:
                self.llm.wait_for_completion()
            except KeyboardInterrupt:
                print("⚡ Force stopping LLM requests...")
        
        # Wait for remaining TTS requests to complete
        if self.tts.is_enabled and not self.tts.is_queue_empty():
            print("⏳ Waiting for remaining TTS requests to complete...")
            try:
                self.tts.wait_for_completion()
            except KeyboardInterrupt:
                print("⚡ Force stopping TTS requests...")
        
        # Stop workers
        self.stt.stop_worker()
        self.llm.stop_worker()
        self.tts.stop_worker()
        
        # Show final statistics
        self._show_final_stats()
        
        # Cleanup ROS2 node
        self.audio_recorder.destroy_node()
        
        self.is_running = False
        print("✅ System shutdown complete")
    
    def _show_final_stats(self):
        """
        Display final system statistics including conversation data.
        """
        audio_stats = self.audio_recorder.get_stats()
        stt_stats = self.stt.get_stats()
        llm_stats = self.llm.get_stats()
        tts_stats = self.tts.get_stats()
        player_stats = self.audio_player.get_stats()
        
        print(f"\n📊 Final System Statistics:")
        print(f"  🎵 Touch Recordings Saved: {audio_stats['segments_saved']}")
        print(f"  📁 Save Directory: {audio_stats['save_directory']}")
        print(f"  🎙️  Audio Frequency: {audio_stats['frequency']} Hz")
        print(f"  📻 Audio Channels: {audio_stats['channels']}")
        print(f"  🤚 Recording Mode: Touch-Triggered")
        print(f"  🤖 STT Enabled: {stt_stats['enabled']}")
        print(f"  🧠 STT Model: {stt_stats['model']}")
        print(f"  📋 STT Remaining Queue: {stt_stats['queue_size']} files")
        print(f"  🤖 LLM Enabled: {llm_stats['enabled']}")
        print(f"  🧠 LLM Model: {llm_stats['model']}")
        print(f"  📋 LLM Remaining Queue: {llm_stats['queue_size']} requests")
        print(f"  ✅ LLM Requests Processed: {llm_stats['requests_processed']}")
        print(f"  ❌ LLM Requests Failed: {llm_stats['requests_failed']}")
        print(f"  🪙 Total Tokens Used: {llm_stats['total_tokens_used']}")
        print(f"  💬 Active Conversations: {llm_stats.get('active_conversations', 0)}")
        print(f"  📝 Active Sessions: {llm_stats.get('active_sessions', 0)}")
        print(f"  💾 Conversations Saved: {llm_stats.get('conversations_saved', 0)}")
        print(f"  📂 Conversations Loaded: {llm_stats.get('conversations_loaded', 0)}")
        print(f"  📁 Save Directory: {llm_stats.get('save_directory', 'N/A')}")
        print(f"  ⏰ Auto-save Delay: {llm_stats.get('auto_save_delay', 0)}s")
        print(f"  🔊 TTS Enabled: {tts_stats['enabled']}")
        print(f"  🎵 TTS Model: {tts_stats['model']}")
        print(f"  🗣️  TTS Voice: {tts_stats['voice']}")
        print(f"  📋 TTS Remaining Queue: {tts_stats['queue_size']} requests")
        print(f"  ✅ TTS Audio Files Generated: {tts_stats['audio_files_generated']}")
        print(f"  📁 TTS Output Directory: {tts_stats['output_dir']}")
        print(f"  🔊 Audio Player Attempts: {player_stats['playback_attempts']}")
        print(f"  ✅ Audio Player Successes: {player_stats['playback_successes']}")
        print(f"  ❌ Audio Player Failures: {player_stats['playback_failures']}")
        print(f"  🎯 Audio Player Success Rate: {player_stats['success_rate']:.1f}%")
        print(f"  📤 SCP Transfers: {player_stats['scp_transfers']}")
        print(f"  ❌ SCP Failures: {player_stats['scp_failures']}")
        print(f"  🎵 qicli Calls: {player_stats['qicli_calls']}")
        print(f"  ❌ qicli Failures: {player_stats['qicli_failures']}")
        print(f"  🎵 Last File Played: {player_stats['last_file_played'] or 'None'}")
        
        # Show conversation summary
        conversation_history = self.get_conversation_history()
        if conversation_history:
            print(f"  💬 Conversation Messages: {len(conversation_history)}")
            user_messages = len([msg for msg in conversation_history if msg['role'] == 'user'])
            ai_messages = len([msg for msg in conversation_history if msg['role'] == 'assistant'])
            print(f"  👤 User Messages: {user_messages}")
            print(f"  🤖 AI Messages: {ai_messages}")
    
    def clear_conversation(self):
        """
        Clear the current conversation history.
        """
        if self.llm.is_enabled:
            self.llm.clear_conversation(self.current_session_id)
            print(f"🗑️  Conversation history cleared for session: {self.current_session_id}")
    
    def get_conversation_history(self) -> list:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        if self.llm.is_enabled:
            return self.llm.get_conversation_history(self.current_session_id)
        return []
    
    def list_saved_conversations(self) -> list:
        """
        List all saved conversation sessions.
        
        Returns:
            List of saved conversation metadata
        """
        if self.llm.is_enabled:
            return self.llm.list_saved_conversations()
        return []
    
    def load_conversation(self, session_id: str) -> bool:
        """
        Load a previously saved conversation and switch to it.
        
        Args:
            session_id: Session identifier to load
            
        Returns:
            True if conversation was loaded successfully
        """
        if self.llm.is_enabled:
            # Try to load the conversation state
            loaded_state = self.llm.load_conversation_state(session_id)
            if loaded_state:
                # Switch to the loaded session
                self.current_session_id = session_id
                print(f"📂 Switched to conversation session: {session_id}")
                return True
            else:
                print(f"❌ Failed to load conversation session: {session_id}")
        return False
    
    def save_current_conversation(self) -> bool:
        """
        Manually save the current conversation state.
        
        Returns:
            True if saved successfully
        """
        if self.llm.is_enabled:
            return self.llm.save_conversation_state(self.current_session_id)
        return False
    
    def delete_conversation(self, session_id: str) -> bool:
        """
        Delete a saved conversation.
        
        Args:
            session_id: Session identifier to delete
            
        Returns:
            True if deleted successfully
        """
        if self.llm.is_enabled:
            return self.llm.delete_conversation_state(session_id)
        return False
    
    def create_new_conversation(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional session identifier (auto-generated if None)
            
        Returns:
            The session identifier of the new conversation
        """
        if session_id is None:
            # Generate a timestamp-based session ID
            session_id = f"nao_session_{int(time.time())}"
        
        # Clear any existing state for this session
        if self.llm.is_enabled:
            self.llm.clear_conversation(session_id)
        
        # Switch to the new session
        self.current_session_id = session_id
        print(f"🆕 Created new conversation session: {session_id}")
        return session_id
    
    def print_conversation_history(self):
        """
        Print the current conversation history to console.
        """
        history = self.get_conversation_history()
        if history:
            print(f"\n💬 Conversation History ({len(history)} messages):")
            print("=" * 60)
            for i, msg in enumerate(history, 1):
                role_icon = "👤" if msg['role'] == 'user' else "🤖"
                timestamp = time.strftime("%H:%M:%S", time.localtime(msg.get('timestamp', 0)))
                print(f"{i:2d}. {role_icon} [{timestamp}] {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
            print("=" * 60)
        else:
            print("💬 No conversation history available")
    
    def get_system_status(self) -> dict:
        """
        Get current system status including conversation state.
        
        Returns:
            Dictionary with system status information
        """
        audio_stats = self.audio_recorder.get_stats()
        stt_stats = self.stt.get_stats()
        llm_stats = self.llm.get_stats()
        tts_stats = self.tts.get_stats()
        
        return {
            'system_running': self.is_running,
            'recording_mode': 'touch_triggered',
            'session_id': self.current_session_id,
            'user_id': self.user_id,
            'conversation_messages': len(self.get_conversation_history()),
            'audio': audio_stats,
            'stt': stt_stats,
            'llm': llm_stats,
            'tts': tts_stats,
            'audio_player': self.audio_player.get_stats()
        }


def signal_handler(signum, frame):
    """
    Handle shutdown signals gracefully.
    """
    print(f"\n📡 Received signal {signum}")
    sys.exit(0)


def main():
    """
    Main entry point for the NAO Touch-Triggered Audio + STT + LLM + TTS system.
    """
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize ROS2
    rclpy.init()
    
    # Create and start the system
    system = None
    try:
        system = NaoAudioSTTSystem()
        system.start()
    except Exception as e:
        print(f"❌ System error: {str(e)}")
    finally:
        if system:
            system.stop()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 