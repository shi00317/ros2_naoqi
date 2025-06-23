import rclpy
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer
from os import path
import threading
import queue
import numpy as np
from typing import Optional, Callable

TOPIC_NAME = "/audio"
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
    for Speech-to-Text services.
    """
    
    def __init__(self, node_name: str = 'nao_audio_streamer'):
        super().__init__(node_name)
        
        # Audio properties
        self.current_frequency = 16000  # Default frequency
        self.channels = 1  # Default to mono
        self.is_streaming = False
        
        # Audio buffer queue for streaming
        self.audio_queue = queue.Queue(maxsize=100)  # Buffer up to 100 audio chunks
        self.streaming_thread = None
        self.streaming_callback: Optional[Callable] = None
        
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
        Processes incoming audio data and adds it to the streaming queue.
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
            
            # Add to queue for streaming (non-blocking)
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
            'queue_size': self.audio_queue.qsize()
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
    Main function to run the audio streamer node.
    """
    rclpy.init(args=args)
    
    # Create audio streamer node
    audio_streamer = NaoAudioStreamer()
    
    try:
        # Example usage: start streaming with a simple callback
        def audio_processor(audio_chunk):
            data = audio_chunk['data']
            freq = audio_chunk['frequency']
            print(f"Received audio chunk: {len(data)} samples at {freq} Hz")
        
        # Start streaming
        audio_streamer.start_streaming(audio_processor)
        
        # Spin the node
        rclpy.spin(audio_streamer)
        
    except KeyboardInterrupt:
        audio_streamer.get_logger().info("Shutting down audio streamer...")
    finally:
        # Clean up
        audio_streamer.stop_streaming()
        audio_streamer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


