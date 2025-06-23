import rclpy
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer, HeadTouch
import threading
import numpy as np
from typing import List, Optional, Callable
import tempfile
import os
import time
from datetime import datetime
from scipy.io import wavfile
from pydub import AudioSegment
from STT import STTTranscriber

TOPIC_NAME = "audio"
HEAD_TOUCH_TOPIC_NAME = "head_touch"
SAVE_DIRECTORY = "/home"


class NaoAudioRecorder(Node):
    """
    ROS2 Audio recorder for NAO robot with touch-triggered recording and STT integration.
    Press any head button to start recording, release to stop and transcribe.
    """
    
    def __init__(self, node_name: str = 'nao_audio_recorder', 
                 on_segment_saved: Optional[Callable[[str], None]] = None,
                 enable_stt: bool = True):
        super().__init__(node_name)
        
        # Audio properties
        self.current_frequency = 16000  # Default frequency
        self.channels = 1  # Default to mono
        
        # Recording properties
        self.recording_buffer: List[np.ndarray] = []
        self.recording_start_time = time.time()
        self.recording_lock = threading.Lock()
        self.segment_counter = 1
        
        # Touch-triggered recording properties
        self.is_recording = False
        
        # STT integration
        self.enable_stt = enable_stt
        self.stt_transcriber = None
        if self.enable_stt:
            try:
                self.stt_transcriber = STTTranscriber()
                self.get_logger().info("✅ STT transcriber initialized")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to initialize STT: {str(e)}")
                self.enable_stt = False
        
        # Callback for when segments are saved
        self.on_segment_saved = on_segment_saved
        
        # Create save directory if it doesn't exist
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        
        # Create subscription to audio topic
        self.audio_subscription = self.create_subscription(
            AudioBuffer,
            TOPIC_NAME,
            self.audio_callback,
            10  # QoS history depth
        )
        
        # Create subscription to head touch topic
        self.head_touch_subscription = self.create_subscription(
            HeadTouch,
            HEAD_TOUCH_TOPIC_NAME,
            self.head_touch_callback,
            10  # QoS history depth
        )
        
        self.get_logger().info(f"🤚 Head touch subscription created for {HEAD_TOUCH_TOPIC_NAME}")
        self.get_logger().info("🎤 Touch-triggered recording mode: Press any head button to start/stop recording")
        self.get_logger().info(f"Audio recorder initialized, saving to {SAVE_DIRECTORY}")

    def head_touch_callback(self, msg: HeadTouch):
        """
        Callback function for head touch subscription.
        Handles touch events to start/stop recording based on state changes only.
        """
        try:
            if msg.state == 1:
                # Any button pressed - start recording
                self._start_recording()
            elif msg.state == 0:
                # Any button released - stop recording and process
                self._stop_recording()
                
        except Exception as e:
            self.get_logger().error(f"Error processing head touch: {str(e)}")

    def _start_recording(self):
        """
        Start recording when any head button is pressed.
        """
        with self.recording_lock:
            if not self.is_recording:
                self.is_recording = True
                self.recording_buffer.clear()
                self.recording_start_time = time.time()
                self.get_logger().info(f"🔴 Started recording - head button pressed")
            else:
                self.get_logger().info(f"⚠️  Already recording")

    def _stop_recording(self):
        """
        Stop recording when any head button is released and process with STT.
        """
        with self.recording_lock:
            if self.is_recording:
                self.is_recording = False
                recording_duration = time.time() - self.recording_start_time
                self.get_logger().info(f"🟥 Stopped recording - head button released ({recording_duration:.1f}s)")
                
                # Save the recorded audio and process with STT
                if self.recording_buffer:
                    output_path = self._save_recording()
                    if output_path and self.enable_stt and self.stt_transcriber:
                        self.get_logger().info("🎙️  Processing with STT...")
                        self.stt_transcriber.add_file_for_transcription(output_path)
                else:
                    self.get_logger().warn("📭 No audio data recorded")
            else:
                self.get_logger().info(f"⚠️  Not currently recording")

    def _save_recording(self) -> Optional[str]:
        """
        Save touch-triggered recording as MP3 file.
        
        Returns:
            Path to saved file, or None if saving failed
        """
        if not self.recording_buffer:
            return None
        
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.recording_buffer)
            self.recording_buffer.clear()
            
            # Generate filename with timestamp and segment number
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nao_touch_recording_{self.segment_counter:04d}_{timestamp}.mp3"
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
                    f"💾 Saved recording {self.segment_counter}: {filename} "
                    f"({duration_seconds:.1f}s, {file_size_mb:.2f}MB)"
                )
                
                # Notify callback that a segment was saved
                if self.on_segment_saved:
                    self.on_segment_saved(output_path)
                
                self.segment_counter += 1
                return output_path
                
            except Exception as e:
                # Clean up temporary WAV file if it exists
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                raise e
                
        except Exception as e:
            self.get_logger().error(f"Error saving recording: {str(e)}")
            return None
    
    def audio_callback(self, msg: AudioBuffer):
        """
        Callback function for audio subscription.
        Only records audio data when a head button is pressed.
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
            
            # Only record when a button is pressed
            with self.recording_lock:
                if self.is_recording:
                    self.recording_buffer.append(audio_data)
            
        except Exception as e:
            self.get_logger().error(f"Error processing audio data: {str(e)}")
    
    def _save_current_segment(self):
        """
        Save current audio buffer as MP3 file and clear buffer.
        This is kept for compatibility but uses the main save method.
        """
        return self._save_recording()
    
    def save_remaining_buffer(self):
        """
        Save any remaining audio in the buffer. Useful when shutting down.
        """
        with self.recording_lock:
            if self.recording_buffer:
                self._save_recording()
    
    def get_stats(self) -> dict:
        """
        Get current recording statistics.
        """
        with self.recording_lock:
            buffer_samples = sum(len(chunk) for chunk in self.recording_buffer)
            buffer_duration = buffer_samples / self.current_frequency if self.current_frequency > 0 else 0
            
        stats = {
            'frequency': self.current_frequency,
            'channels': self.channels,
            'segments_saved': self.segment_counter - 1,
            'current_buffer_duration_seconds': buffer_duration,
            'save_directory': SAVE_DIRECTORY,
            'is_recording': self.is_recording,
            'stt_enabled': self.enable_stt
        }
        
        # Add STT stats if available
        if self.enable_stt and self.stt_transcriber:
            stt_stats = self.stt_transcriber.get_stats()
            stats['stt_stats'] = stt_stats
            
        return stats


def main(args=None):
    """
    Main function to run the touch-triggered audio recorder with STT.
    """
    rclpy.init(args=args)
    
    # Create touch-triggered audio recorder with STT
    audio_recorder = NaoAudioRecorder(enable_stt=True)
    
    try:
        print(f"🎤 Starting NAO Touch-Triggered Audio Recorder with STT")
        print(f"🤚 Touch any head button to start recording, release to stop and transcribe")
        print(f"📁 Saving recordings to: {SAVE_DIRECTORY}")
        print(f"⏹️  Press Ctrl+C to stop\n")
        
        # Spin the node
        rclpy.spin(audio_recorder)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down touch-triggered audio recorder...")
    finally:
        # Save any remaining audio in buffer
        audio_recorder.save_remaining_buffer()
        
        # Stop STT worker if enabled
        if audio_recorder.enable_stt and audio_recorder.stt_transcriber:
            audio_recorder.stt_transcriber.stop_worker()
        
        # Show final stats
        stats = audio_recorder.get_stats()
        print(f"\n📊 Final stats:")
        print(f"  Recordings saved: {stats['segments_saved']}")
        print(f"  STT enabled: {stats['stt_enabled']}")
        print(f"  Save directory: {stats['save_directory']}")
        
        if 'stt_stats' in stats:
            stt_stats = stats['stt_stats']
            print(f"  STT queue size: {stt_stats.get('queue_size', 'N/A')}")
        
        audio_recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


