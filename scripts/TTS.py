#!/usr/bin/env python3

import os
import threading
import time
import subprocess
from pathlib import Path
from typing import Optional, Callable
from queue import Queue, Empty
from openai import OpenAI


class TTSProcessor:
    """
    Text-to-Speech processor using OpenAI's TTS API to convert text to speech.
    Supports queue-based processing and NAO robot audio playback.
    """
    
    def __init__(self, 
                 model: str = "tts-1",
                 voice: str = "nova",
                 output_dir: str = "/tmp/nao_tts",
                 api_key: Optional[str] = None,
                 max_queue_size: int = 10):
        """
        Initialize the TTS processor.
        
        Args:
            model: OpenAI TTS model to use ("tts-1" or "tts-1-hd")
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            output_dir: Directory to save TTS audio files
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
            max_queue_size: Maximum number of requests to queue
        """
        self.model = model
        self.voice = voice
        self.output_dir = Path(output_dir)
        self.max_queue_size = max_queue_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
        
        # Processing queue and worker thread
        self.request_queue = Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.is_enabled = True
        self.is_running = False
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'requests_failed': 0,
            'audio_files_generated': 0,
            'requests_queued': 0
        }
        
        print(f"🔊 TTS Processor initialized with model: {self.model}, voice: {self.voice}")
    
    def start_worker(self):
        """
        Start the background worker thread for processing TTS requests.
        """
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("🚀 TTS worker thread started")
    
    def stop_worker(self):
        """
        Stop the background worker thread.
        """
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        print("🛑 TTS worker thread stopped")
    
    def add_request(self, text: str, 
                   callback: Optional[Callable[[str], None]] = None,
                   save_file: bool = True):
        """
        Add text for TTS processing.
        
        Args:
            text: Text to convert to speech
            callback: Optional callback function to call with the audio file path
            save_file: Whether to save the audio file
        """
        if not self.is_enabled:
            print("⚠️  TTS processing is disabled")
            return
        
        if not text.strip():
            print("⚠️  Empty text provided for TTS processing")
            return
        
        try:
            request = {
                'text': text.strip(),
                'callback': callback,
                'save_file': save_file,
                'timestamp': time.time()
            }
            self.request_queue.put(request, block=False)
            self.stats['requests_queued'] += 1
            print(f"🎵 Added text to TTS queue: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        except:
            print(f"⚠️  TTS queue is full, dropping request")
    
    def _worker_loop(self):
        """
        Background worker loop for processing TTS requests.
        """
        print("🔄 TTS worker loop started")
        
        while self.is_running:
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=1.0)
                self._process_request(request)
                self.request_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Error in TTS worker loop: {str(e)}")
        
        print("🔄 TTS worker loop stopped")
    
    def _process_request(self, request: dict):
        """
        Process a single TTS request.
        
        Args:
            request: Request dictionary containing text and options
        """
        try:
            text = request['text']
            callback = request.get('callback')
            save_file = request['save_file']
            
            print(f"🎤 Processing TTS request: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate unique filename based on timestamp
            timestamp = int(time.time() * 1000)
            audio_filename = f"tts_speech_{timestamp}.mp3"
            audio_file_path = self.output_dir / audio_filename
            
            # Create TTS audio using OpenAI API
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3"
            ) as response:
                if save_file:
                    response.stream_to_file(audio_file_path)
                    self.stats['audio_files_generated'] += 1
                    print(f"🎵 TTS audio saved: {audio_file_path}")
                else:
                    # Still need to consume the response
                    audio_content = response.content
            
            # Update statistics
            self.stats['requests_processed'] += 1
            
            # Call callback if provided
            if callback:
                callback(str(audio_file_path) if save_file else None)
            
        except Exception as e:
            self.stats['requests_failed'] += 1
            error_msg = f"❌ TTS processing failed: {str(e)}"
            print(error_msg)
            
            # Call callback with error if provided
            callback = request.get('callback')
            if callback:
                callback(f"Error: {str(e)}")
    

    def synthesize_audio(self, text: str) -> Optional[str]:
        """
        Synchronously synthesize text to speech.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Path to the generated audio file, or None if failed
        """
        if not self.is_enabled:
            print("❌ TTS not enabled")
            return None
        
        if not text.strip():
            print("❌ Empty text provided")
            return None
        
        try:
            print(f"🎤 Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            audio_filename = f"tts_speech_sync_{timestamp}.mp3"
            audio_file_path = self.output_dir / audio_filename
            
            # Create TTS audio
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3"
            ) as response:
                response.stream_to_file(audio_file_path)
            
            self.stats['audio_files_generated'] += 1
            print(f"🎵 TTS audio saved: {audio_file_path}")
            
            return str(audio_file_path)
            
        except Exception as e:
            print(f"❌ Synchronous TTS failed: {str(e)}")
            return None
    
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
            print(f"⏰ Timeout waiting for TTS queue completion")
        else:
            print(f"✅ All TTS requests processed")
    
    def enable(self):
        """Enable TTS processing."""
        self.is_enabled = True
        if not self.is_running:
            self.start_worker()
        print("✅ TTS processing enabled")
    
    def disable(self):
        """Disable TTS processing."""
        self.is_enabled = False
        print("⏸️  TTS processing disabled")
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old TTS audio files.
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            removed_count = 0
            for audio_file in self.output_dir.glob("tts_speech_*.mp3"):
                if current_time - audio_file.stat().st_mtime > max_age_seconds:
                    audio_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                print(f"🧹 Cleaned up {removed_count} old TTS files")
                
        except Exception as e:
            print(f"⚠️  Error cleaning up TTS files: {str(e)}")
    
    def get_stats(self) -> dict:
        """
        Get TTS processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'enabled': self.is_enabled,
            'model': self.model,
            'voice': self.voice,
            'output_dir': str(self.output_dir),
            'queue_size': self.get_queue_size(),
            'requests_processed': self.stats['requests_processed'],
            'requests_failed': self.stats['requests_failed'],
            'audio_files_generated': self.stats['audio_files_generated'],
            'requests_queued': self.stats['requests_queued']
        }


if __name__ == '__main__':
    # Test the TTS processor
    tts = TTSProcessor()
    
    def print_result(file_path: str):
        print(f"🎵 TTS Result: {file_path}")
    
    tts.start_worker()
    
    # Test request
    tts.add_request("Today is a wonderful day to build something people love!", print_result)
    
    # Wait for completion
    tts.wait_for_completion()
    tts.stop_worker()
    
    print(f"📊 Final stats: {tts.get_stats()}")
