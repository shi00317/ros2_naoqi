#!/usr/bin/env python3

import rclpy
import signal
import sys
from Audio import NaoAudioRecorder, SAVE_DIRECTORY
from STT import STTTranscriber
from LLM import LLMProcessor
from TTS import TTSProcessor
from AudioPlayer import NaoAudioPlayer


class NaoAudioSTTSystem:
    """
    Main system that coordinates touch-triggered audio recording, speech-to-text transcription, 
    LLM processing, and text-to-speech synthesis.
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
            model="tts-1",
            voice="nova",
            output_dir="/tmp/nao_tts",
            max_queue_size=10
        )
        
        # Initialize LLM processor with TTS callback
        self.llm = LLMProcessor(
            model="gpt-4o-mini",
            max_queue_size=10,
            system_prompt="You are a helpful assistant for a NAO robot. Respond to user speech in a natural, friendly, and concise manner. Keep responses brief but informative."
        )
        
        # Initialize STT transcriber with LLM callback
        self.stt = STTTranscriber(
            model="whisper-1", 
            max_queue_size=20,
            on_transcription_complete=self._on_transcription_complete
        )
        
        # Initialize touch-triggered audio recorder with callback to STT
        self.audio_recorder = NaoAudioRecorder(
            node_name='nao_audio_recorder',
            on_segment_saved=self._on_audio_segment_saved,
            enable_stt=True
        )
        
        # System state
        self.is_running = False
        
        print("✅ NAO Touch-Triggered Audio + STT + LLM + TTS System initialized")
    
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
        Sends the transcription to LLM for processing.
        
        Args:
            transcription: The transcribed text
        """
        if self.llm.is_enabled and transcription.strip():
            self.llm.add_request(transcription, self._on_llm_response)
    
    def _on_llm_response(self, response: str):
        """
        Callback function called when LLM processing is complete.
        Prints the LLM response and converts it to speech.
        
        Args:
            response: The LLM response text
        """
        print(f"\n🤖 LLM RESPONSE: {response}")
        print("═" * 60)
        print()
        
        # Convert LLM response to speech and play on NAO
        if self.tts.is_enabled and response.strip() and not response.startswith("Error:"):
            self.tts.add_request(response, self._on_tts_complete, save_file=True)
    
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
        print(f"📁 Audio recordings saved to: {SAVE_DIRECTORY}")
        print(f"🤖 STT Status: {'Enabled' if self.stt.is_enabled else 'Disabled'}")
        print(f"🧠 STT Model: {self.stt.model}")
        print(f"📋 STT Queue Size: {self.stt.get_queue_size()}")
        print(f"🤖 LLM Status: {'Enabled' if self.llm.is_enabled else 'Disabled'}")
        print(f"🧠 LLM Model: {self.llm.model}")
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
        Display final system statistics.
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
    
    def get_system_status(self) -> dict:
        """
        Get current system status.
        
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