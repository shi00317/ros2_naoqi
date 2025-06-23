import rclpy
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer
import threading
import numpy as np
from typing import List
import tempfile
import os
import time
from datetime import datetime
from scipy.io import wavfile
from pydub import AudioSegment
from openai import OpenAI

TOPIC_NAME = "audio"
SAVE_DIRECTORY = "/home"
SEGMENT_DURATION_SECONDS = 10


class SimpleNaoAudioRecorder(Node):
    """
    Simplified ROS2 Audio recorder for NAO robot that saves 10-second MP3 segments to /home directory
    with OpenAI STT transcription.
    """
    
    def __init__(self, node_name: str = 'simple_nao_audio_recorder', enable_stt: bool = True):
        super().__init__(node_name)
        
        # Audio properties
        self.current_frequency = 16000  # Default frequency
        self.channels = 1  # Default to mono
        
        # Recording properties
        self.recording_buffer: List[np.ndarray] = []
        self.recording_start_time = time.time()
        self.recording_lock = threading.Lock()
        self.segment_counter = 1
        
        # OpenAI STT integration
        self.enable_stt = enable_stt
        self.openai_client = None
        self.transcription_model = "whisper-1"  # OpenAI Whisper model
        
        if self.enable_stt:
            try:
                self.openai_client = OpenAI()
                self.get_logger().info("OpenAI client initialized successfully")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize OpenAI client: {str(e)}")
                self.enable_stt = False
        
        # Create save directory if it doesn't exist
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        
        # Create subscription to audio topic
        self.audio_subscription = self.create_subscription(
            AudioBuffer,
            TOPIC_NAME,
            self.audio_callback,
            10  # QoS history depth
        )
        
        stt_status = "with OpenAI STT" if self.enable_stt else "without STT"
        self.get_logger().info(f"Simple audio recorder initialized, saving to {SAVE_DIRECTORY} {stt_status}")
        self.get_logger().info(f"Recording {SEGMENT_DURATION_SECONDS}s segments from {TOPIC_NAME}")
    
    def audio_callback(self, msg: AudioBuffer):
        """
        Callback function for audio subscription.
        Processes incoming audio data and saves 10-second segments.
        """
        try:
            # Update audio properties from message
            self.current_frequency = msg.frequency
            self.channels = len(msg.channel_map) if msg.channel_map else 1
            
            # Convert int16 data to numpy array
            audio_data = np.array(msg.data, dtype=np.int16)
            
            # Reshape data based on channel configuration
            if self.channels > 1 and len(audio_data) > 0:
                # Deinterlace multi-channel audio - use first channel only
                samples_per_channel = len(audio_data) // self.channels
                audio_data = audio_data.reshape((samples_per_channel, self.channels))
                audio_data = audio_data[:, 0]
            
            # Add to recording buffer
            with self.recording_lock:
                self.recording_buffer.append(audio_data)
                
                # Check if 10 seconds have passed
                current_time = time.time()
                if current_time - self.recording_start_time >= SEGMENT_DURATION_SECONDS:
                    self._save_current_segment()
                    self.recording_start_time = current_time
            
        except Exception as e:
            self.get_logger().error(f"Error processing audio data: {str(e)}")
    
    def _save_current_segment(self):
        """
        Save current audio buffer as MP3 file and clear buffer.
        """
        if not self.recording_buffer:
            return
        
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.recording_buffer)
            self.recording_buffer.clear()
            
            # Generate filename with timestamp and segment number
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nao_audio_segment_{self.segment_counter:04d}_{timestamp}.mp3"
            output_path = os.path.join(SAVE_DIRECTORY, filename)
            
            # Create temporary WAV file for conversion
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav', prefix='nao_temp_')
            os.close(temp_wav_fd)
            
            try:
                # Save as temporary WAV file first
                wavfile.write(temp_wav_path, self.current_frequency, audio_data)
                
                # Convert WAV to MP3 using pydub
                audio_segment = AudioSegment.from_wav(temp_wav_path)
                audio_segment.export(output_path, format="mp3", bitrate="128k")
                
                # Clean up temporary WAV file
                os.remove(temp_wav_path)
                
                # Log success
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                duration_seconds = len(audio_data) / self.current_frequency
                self.get_logger().info(
                    f"Saved segment {self.segment_counter}: {filename} "
                    f"({duration_seconds:.1f}s, {file_size_mb:.2f}MB)"
                )
                
                # Transcribe the saved audio file if STT is enabled
                if self.enable_stt:
                    self._transcribe_and_print(output_path)
                
                self.segment_counter += 1
                
            except Exception as e:
                # Clean up temporary WAV file if it exists
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                raise e
                
        except Exception as e:
            self.get_logger().error(f"Error saving audio segment: {str(e)}")
    
    def transcribe_audio_file(self, file_path: str) -> str:
        """
        Transcribe an audio file using OpenAI's Whisper STT model.
        
        Args:
            file_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text, or empty string if transcription failed
        """
        if not self.enable_stt or not self.openai_client:
            self.get_logger().error("OpenAI STT not available")
            return ""
        
        if not os.path.exists(file_path):
            self.get_logger().error(f"Audio file not found: {file_path}")
            return ""
        
        try:
            self.get_logger().info(f"Transcribing audio file: {file_path}")
            
            with open(file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=self.transcription_model,
                    file=audio_file,
                    response_format="text"
                )
            
            return transcription.strip() if transcription else ""
            
        except Exception as e:
            self.get_logger().error(f"OpenAI transcription failed: {str(e)}")
            return ""
    
    def _transcribe_and_print(self, file_path: str):
        """
        Transcribe an audio file and print the result to terminal.
        Runs in a separate thread to avoid blocking audio processing.
        
        Args:
            file_path: Path to the audio file to transcribe
        """
        def transcribe_worker():
            try:
                transcription = self.transcribe_audio_file(file_path)
                if transcription:
                    # Print to terminal with clear formatting
                    print(f"\n🎙️  TRANSCRIPTION: {transcription}")
                    print("─" * 50)  # Visual separator
                    self.get_logger().info(f"Transcribed: {transcription}")
                else:
                    print(f"\n🔇 No speech detected in audio segment")
                    self.get_logger().warn("Transcription returned empty result")
            except Exception as e:
                self.get_logger().error(f"Error in transcription worker: {str(e)}")
        
        # Run transcription in a separate thread to avoid blocking audio processing
        transcription_thread = threading.Thread(target=transcribe_worker, daemon=True)
        transcription_thread.start()
    
    def set_transcription_model(self, model: str):
        """
        Set the OpenAI transcription model to use.
        
        Args:
            model: Model name (e.g., 'whisper-1')
        """
        self.transcription_model = model
        self.get_logger().info(f"Transcription model set to: {model}")
    
    def toggle_stt(self, enable: bool):
        """
        Enable or disable STT transcription.
        
        Args:
            enable: True to enable STT, False to disable
        """
        if enable and not self.openai_client:
            try:
                self.openai_client = OpenAI()
                self.enable_stt = True
                self.get_logger().info("STT enabled")
            except Exception as e:
                self.get_logger().error(f"Failed to enable STT: {str(e)}")
                self.enable_stt = False
        else:
            self.enable_stt = enable
            status = "enabled" if enable else "disabled"
            self.get_logger().info(f"STT {status}")
    
    def get_stats(self) -> dict:
        """
        Get current recording statistics.
        """
        with self.recording_lock:
            buffer_samples = sum(len(chunk) for chunk in self.recording_buffer)
            buffer_duration = buffer_samples / self.current_frequency if self.current_frequency > 0 else 0
            
        return {
            'frequency': self.current_frequency,
            'channels': self.channels,
            'segments_saved': self.segment_counter - 1,
            'current_buffer_duration_seconds': buffer_duration,
            'save_directory': SAVE_DIRECTORY,
            'stt_enabled': self.enable_stt,
            'transcription_model': self.transcription_model
        }


def main(args=None):
    """
    Main function to run the simple audio recorder with STT.
    """
    rclpy.init(args=args)
    
    # Create simple audio recorder with STT enabled
    audio_recorder = SimpleNaoAudioRecorder(enable_stt=True)
    
    try:
        print(f"🎤 Starting Simple NAO Audio Recorder with STT")
        print(f"📁 Saving {SEGMENT_DURATION_SECONDS}s MP3 segments to: {SAVE_DIRECTORY}")
        print(f"🤖 OpenAI STT: {'Enabled' if audio_recorder.enable_stt else 'Disabled'}")
        print(f"⏹️  Press Ctrl+C to stop\n")
        
        # Spin the node
        rclpy.spin(audio_recorder)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down audio recorder...")
    finally:
        # Save any remaining audio in buffer
        with audio_recorder.recording_lock:
            if audio_recorder.recording_buffer:
                audio_recorder._save_current_segment()
        
        # Show final stats
        stats = audio_recorder.get_stats()
        print(f"\n📊 Final stats:")
        print(f"  Segments saved: {stats['segments_saved']}")
        print(f"  Save directory: {stats['save_directory']}")
        print(f"  STT enabled: {stats['stt_enabled']}")
        
        audio_recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


