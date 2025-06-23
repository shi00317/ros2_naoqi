import rclpy
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer
from os import path
import threading
import queue
import numpy as np
from typing import Optional, Callable, List
import tempfile
import os
import time
from scipy.io import wavfile
import math
from openai import OpenAI

TOPIC_NAME = "/nao_robot/microphone/naoqi_microphone/audio_raw"
'''
# timestanp the audio buffer
std_msgs/Header header
# current frequency of the audio interface
uint16 frequency
# channel order properties :
uint8 CHANNEL_FRONT_LEFT=0
uint8 CHANNEL_FRONT_CENTER=1
uint8 CHANNEL_FRONT_RIGHT=2
uint8 CHANNEL_REAR_LEFT=3
uint8 CHANNEL_REAR_CENTER=4
uint8 CHANNEL_REAR_RIGHT=5
uint8 CHANNEL_SURROUND_LEFT=6
uint8 CHANNEL_SURROUND_RIGHT=7
uint8 CHANNEL_SUBWOOFER=8
uint8 CHANNEL_LFE=9
# channel order of the current buffer
uint8[] channel_map
# interlaced data of the audio buffer
int16[] data
'''


class NaoAudioStreamer(Node):
    """
    ROS2 Audio class for subscribing to NAO robot audio and providing streaming functionality
    for Speech-to-Text services. Supports both real-time streaming and file-based recording
    with OpenAI transcription integration.
    """
    
    def __init__(self, node_name: str = 'nao_audio_streamer', enable_openai: bool = True):
        super().__init__(node_name)
        
        # Audio properties
        self.current_frequency = 16000  # Default frequency
        self.channels = 1  # Default to mono
        self.is_streaming = False
        self.is_recording = False
        
        # Audio buffer queue for streaming
        self.audio_queue = queue.Queue(maxsize=100)  # Buffer up to 100 audio chunks
        self.streaming_thread = None
        self.streaming_callback: Optional[Callable] = None
        
        # File recording properties
        self.recording_buffer: List[np.ndarray] = []
        self.recording_start_time = None
        self.max_file_size_mb = 20  # Stay under 25MB limit with some margin
        self.temp_files: List[str] = []  # Track temporary files for cleanup
        self.recording_lock = threading.Lock()
        
        # OpenAI STT integration
        self.enable_openai = enable_openai
        self.openai_client = None
        self.auto_transcribe = False
        self.transcription_model = "gpt-4o-mini-transcribe"  # Default OpenAI model
        
        if self.enable_openai:
            try:
                self.openai_client = OpenAI()
                self.get_logger().info("OpenAI client initialized successfully")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize OpenAI client: {str(e)}")
                self.enable_openai = False
        
        # Create subscription to audio topic
        self.audio_subscription = self.create_subscription(
            AudioBuffer,
            TOPIC_NAME,
            self.audio_callback,
            10  # QoS history depth
        )
        
        self.get_logger().info(f"Audio streamer initialized, subscribing to {TOPIC_NAME}")
    
    def audio_callback(self, msg: AudioBuffer):
        """
        Callback function for audio subscription.
        Processes incoming audio data and adds it to streaming queue and recording buffer.
        """
        try:
            # Update audio properties from message
            self.current_frequency = msg.frequency
            self.channels = len(msg.channel_map) if msg.channel_map else 1
            
            # Convert int16 data to numpy array
            audio_data = np.array(msg.data, dtype=np.int16)
            
            # Reshape data based on channel configuration
            if self.channels > 1 and len(audio_data) > 0:
                # Deinterlace multi-channel audio
                samples_per_channel = len(audio_data) // self.channels
                audio_data = audio_data.reshape((samples_per_channel, self.channels))
                # For STT, typically use only the first channel
                audio_data = audio_data[:, 0]
            
            # Add to streaming queue (non-blocking)
            if self.is_streaming:
                try:
                    self.audio_queue.put_nowait({
                        'data': audio_data,
                        'frequency': self.current_frequency,
                        'timestamp': msg.header.stamp,
                        'channels': self.channels
                    })
                except queue.Full:
                    # Remove oldest item if queue is full
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait({
                            'data': audio_data,
                            'frequency': self.current_frequency,
                            'timestamp': msg.header.stamp,
                            'channels': self.channels
                        })
                    except queue.Empty:
                        pass
            
            # Add to recording buffer if recording
            if self.is_recording:
                with self.recording_lock:
                    self.recording_buffer.append(audio_data)
                    
                    # Check if we need to save due to size limit
                    current_size = self._estimate_buffer_size_mb()
                    if current_size >= self.max_file_size_mb:
                        saved_file = self._save_current_buffer()
                        # Auto-transcribe if enabled
                        if self.auto_transcribe and saved_file:
                            self._transcribe_and_print(saved_file)
            
        except Exception as e:
            self.get_logger().error(f"Error processing audio data: {str(e)}")
    
    def start_streaming(self, callback_function: Optional[Callable] = None):
        """
        Start audio streaming. Audio data will be available through the callback function
        or can be retrieved using get_audio_chunk().
        
        Args:
            callback_function: Optional callback function that will be called with each audio chunk
        """
        if self.is_streaming:
            self.get_logger().warn("Audio streaming is already active")
            return
        
        self.is_streaming = True
        self.streaming_callback = callback_function
        
        if callback_function:
            # Start streaming thread if callback is provided
            self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
            self.streaming_thread.start()
        
        self.get_logger().info("Audio streaming started")
    
    def stop_streaming(self):
        """
        Stop audio streaming and clean up resources.
        """
        self.is_streaming = False
        self.streaming_callback = None
        
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=1.0)
        
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        self.get_logger().info("Audio streaming stopped")
    
    def start_recording(self, max_file_size_mb: float = 20, auto_transcribe: bool = True):
        """
        Start recording audio to files for STT service upload.
        
        Args:
            max_file_size_mb: Maximum file size in MB (default 20MB to stay under 25MB limit)
            auto_transcribe: Whether to automatically transcribe saved audio segments
        """
        if self.is_recording:
            self.get_logger().warn("Audio recording is already active")
            return
        
        self.max_file_size_mb = max_file_size_mb
        self.auto_transcribe = auto_transcribe and self.enable_openai
        self.is_recording = True
        self.recording_start_time = time.time()
        
        with self.recording_lock:
            self.recording_buffer.clear()
        
        transcribe_msg = " with auto-transcription" if self.auto_transcribe else ""
        self.get_logger().info(f"Audio recording started (max file size: {max_file_size_mb}MB){transcribe_msg}")
    
    def stop_recording(self, save_final_segment: bool = True) -> Optional[str]:
        """
        Stop recording and optionally save the final audio segment.
        
        Args:
            save_final_segment: Whether to save the remaining audio in buffer
            
        Returns:
            Path to the final saved file, or None if no final segment
        """
        if not self.is_recording:
            self.get_logger().warn("Audio recording is not active")
            return None
        
        self.is_recording = False
        final_file = None
        
        if save_final_segment:
            with self.recording_lock:
                if self.recording_buffer:
                    final_file = self._save_current_buffer()
                    # Auto-transcribe final segment if enabled
                    if self.auto_transcribe and final_file:
                        self._transcribe_and_print(final_file)
        
        self.auto_transcribe = False
        self.get_logger().info("Audio recording stopped")
        return final_file
    
    def save_audio_segment(self, duration_seconds: Optional[float] = None, transcribe: bool = None) -> Optional[str]:
        """
        Manually save current audio buffer to a file.
        
        Args:
            duration_seconds: If specified, save only the last N seconds of audio
            transcribe: Whether to transcribe this segment (uses auto_transcribe setting if None)
            
        Returns:
            Path to the saved file, or None if no audio to save
        """
        if not self.is_recording:
            self.get_logger().warn("Cannot save segment: recording is not active")
            return None
        
        with self.recording_lock:
            if not self.recording_buffer:
                self.get_logger().warn("No audio data to save")
                return None
            
            buffer_to_save = self.recording_buffer.copy()
            
            # If duration is specified, trim the buffer
            if duration_seconds:
                samples_needed = int(duration_seconds * self.current_frequency)
                total_samples = sum(len(chunk) for chunk in buffer_to_save)
                
                if total_samples > samples_needed:
                    # Keep only the last N seconds
                    samples_to_remove = total_samples - samples_needed
                    while samples_to_remove > 0 and buffer_to_save:
                        chunk = buffer_to_save[0]
                        if len(chunk) <= samples_to_remove:
                            samples_to_remove -= len(chunk)
                            buffer_to_save.pop(0)
                        else:
                            buffer_to_save[0] = chunk[samples_to_remove:]
                            break
            
            # Save and clear the specified portion
            if duration_seconds:
                # Don't clear buffer for manual saves
                saved_file = self._save_buffer_to_file(buffer_to_save)
            else:
                # Clear buffer for full saves
                saved_file = self._save_current_buffer()
            
            # Transcribe if requested
            should_transcribe = transcribe if transcribe is not None else self.auto_transcribe
            if should_transcribe and saved_file:
                self._transcribe_and_print(saved_file)
            
            return saved_file
    
    def transcribe_file(self, file_path: str, model: str = None) -> Optional[str]:
        """
        Transcribe an audio file using OpenAI's API.
        
        Args:
            file_path: Path to the audio file
            model: OpenAI model to use (defaults to self.transcription_model)
            
        Returns:
            Transcribed text, or None if transcription failed
        """
        if not self.enable_openai or not self.openai_client:
            self.get_logger().error("OpenAI client not available for transcription")
            return None
        
        if not os.path.exists(file_path):
            self.get_logger().error(f"Audio file not found: {file_path}")
            return None
        
        try:
            model = model or self.transcription_model
            
            with open(file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=model,
                    file=audio_file
                )
            
            return transcription.text
            
        except Exception as e:
            self.get_logger().error(f"OpenAI transcription failed: {str(e)}")
            return None
    
    def _transcribe_and_print(self, file_path: str):
        """
        Transcribe an audio file and print the result.
        
        Args:
            file_path: Path to the audio file to transcribe
        """
        def transcribe_worker():
            try:
                transcription = self.transcribe_file(file_path)
                if transcription:
                    print(f"\n🎙️  USER SPEECH: {transcription}\n")
                    self.get_logger().info(f"Transcribed: {transcription}")
                else:
                    self.get_logger().warn("Transcription returned empty result")
            except Exception as e:
                self.get_logger().error(f"Error in transcription worker: {str(e)}")
        
        # Run transcription in a separate thread to avoid blocking
        transcription_thread = threading.Thread(target=transcribe_worker, daemon=True)
        transcription_thread.start()
    
    def get_recorded_files(self) -> List[str]:
        """
        Get list of all recorded audio files.
        
        Returns:
            List of file paths
        """
        return self.temp_files.copy()
    
    def cleanup_temp_files(self):
        """
        Delete all temporary audio files.
        """
        deleted_count = 0
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                self.get_logger().error(f"Error deleting {file_path}: {str(e)}")
        
        self.temp_files.clear()
        self.get_logger().info(f"Cleaned up {deleted_count} temporary audio files")
    
    def set_transcription_model(self, model: str):
        """
        Set the OpenAI transcription model to use.
        
        Args:
            model: Model name (e.g., 'whisper-1')
        """
        self.transcription_model = model
        self.get_logger().info(f"Transcription model set to: {model}")
    
    def _estimate_buffer_size_mb(self) -> float:
        """
        Estimate the size of current recording buffer in MB.
        """
        if not self.recording_buffer:
            return 0.0
        
        total_samples = sum(len(chunk) for chunk in self.recording_buffer)
        # int16 = 2 bytes per sample, plus WAV header overhead
        estimated_bytes = total_samples * 2 + 1024  # Add some header overhead
        return estimated_bytes / (1024 * 1024)
    
    def _save_current_buffer(self) -> Optional[str]:
        """
        Save current recording buffer to file and clear it.
        """
        if not self.recording_buffer:
            return None
        
        buffer_to_save = self.recording_buffer.copy()
        self.recording_buffer.clear()
        
        return self._save_buffer_to_file(buffer_to_save)
    
    def _save_buffer_to_file(self, buffer: List[np.ndarray]) -> Optional[str]:
        """
        Save audio buffer to a WAV file.
        
        Args:
            buffer: List of numpy arrays containing audio data
            
        Returns:
            Path to the saved file
        """
        if not buffer:
            return None
        
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(buffer)
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='nao_audio_')
            os.close(temp_fd)  # Close the file descriptor, we'll write with scipy
            
            # Save as WAV file
            wavfile.write(temp_path, self.current_frequency, audio_data)
            
            # Track the file for cleanup
            self.temp_files.append(temp_path)
            
            file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            self.get_logger().info(
                f"Saved audio segment: {temp_path} "
                f"({len(audio_data)} samples, {file_size_mb:.2f}MB)"
            )
            
            return temp_path
            
        except Exception as e:
            self.get_logger().error(f"Error saving audio buffer: {str(e)}")
            return None
    
    def _streaming_worker(self):
        """
        Worker thread that processes audio chunks and calls the streaming callback.
        """
        while self.is_streaming and self.streaming_callback:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.streaming_callback(audio_chunk)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in streaming worker: {str(e)}")
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Get the next available audio chunk from the queue.
        
        Args:
            timeout: Maximum time to wait for audio chunk (seconds)
            
        Returns:
            Dictionary containing audio data and metadata, or None if timeout
        """
        if not self.is_streaming:
            self.get_logger().warn("Audio streaming is not active")
            return None
        
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_audio_properties(self) -> dict:
        """
        Get current audio properties.
        
        Returns:
            Dictionary containing frequency, channels, and streaming status
        """
        return {
            'frequency': self.current_frequency,
            'channels': self.channels,
            'is_streaming': self.is_streaming,
            'is_recording': self.is_recording,
            'auto_transcribe': self.auto_transcribe,
            'openai_enabled': self.enable_openai,
            'transcription_model': self.transcription_model,
            'queue_size': self.audio_queue.qsize(),
            'recording_buffer_size_mb': self._estimate_buffer_size_mb(),
            'recorded_files_count': len(self.temp_files)
        }
    
    def clear_audio_buffer(self):
        """
        Clear the audio buffer queue.
        """
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.get_logger().info("Audio buffer cleared")


def main(args=None):
    """
    Main function to run the audio streamer node with OpenAI transcription.
    """
    rclpy.init(args=args)
    
    # Create audio streamer node with OpenAI integration
    audio_streamer = NaoAudioStreamer(enable_openai=True)
    
    try:
        print("🎤 Starting NAO Audio Streamer with OpenAI Transcription")
        print("📝 User speech will be automatically transcribed and displayed")
        print("⏹️  Press Ctrl+C to stop\n")
        
        # Start recording with auto-transcription enabled
        audio_streamer.start_recording(max_file_size_mb=20, auto_transcribe=True)
        
        # Optional: also start streaming for real-time monitoring
        def audio_monitor(audio_chunk):
            data = audio_chunk['data']
            # Show audio activity indicator
            volume = np.sqrt(np.mean(data**2))
            if volume > 1000:  # Adjust threshold as needed
                print("🔊", end="", flush=True)
        
        audio_streamer.start_streaming(audio_monitor)
        
        # Spin the node
        rclpy.spin(audio_streamer)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down audio streamer...")
    finally:
        # Clean up
        audio_streamer.stop_streaming()
        final_file = audio_streamer.stop_recording()
        
        # Show recorded files
        files = audio_streamer.get_recorded_files()
        print(f"\n📁 Recorded {len(files)} audio files:")
        for file_path in files:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file_path} ({size_mb:.2f}MB)")
        
        # Cleanup temporary files
        audio_streamer.cleanup_temp_files()
        
        audio_streamer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


