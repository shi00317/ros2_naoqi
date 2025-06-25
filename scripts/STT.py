#!/usr/bin/env python3

import os
import threading
import queue
from typing import Optional
from openai import OpenAI
from prompt import STTPrompt

class STTTranscriber:
    """
    Speech-to-Text transcriber using OpenAI's Whisper API.
    Handles transcription of audio files with queue-based processing.
    """
    
    def __init__(self, model: str = "whisper-1", max_queue_size: int = 10, on_transcription_complete=None):
        """
        Initialize the STT transcriber.
        
        Args:
            model: OpenAI model to use for transcription
            max_queue_size: Maximum number of files to queue for transcription
            on_transcription_complete: Callback function called when transcription is complete
        """
        self.model = model
        self.openai_client = None
        self.is_enabled = False
        self.on_transcription_complete = on_transcription_complete
        
        # Queue for files to transcribe
        self.transcription_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.is_running = False
        
        # Initialize OpenAI client
        self._initialize_client()
        
        # Start worker thread if client is available
        if self.is_enabled:
            self.start_worker()
    
    def _initialize_client(self):
        """
        Initialize the OpenAI client.
        """
        try:
            self.openai_client = OpenAI()
            self.is_enabled = True
            print("✅ OpenAI STT client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI client: {str(e)}")
            self.is_enabled = False
    
    def start_worker(self):
        """
        Start the background worker thread for processing transcriptions.
        """
        if not self.is_enabled:
            print("⚠️  Cannot start STT worker: OpenAI client not available")
            return
        
        if self.is_running:
            print("⚠️  STT worker is already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.worker_thread.start()
        print("🚀 STT worker thread started")
    
    def stop_worker(self):
        """
        Stop the background worker thread.
        """
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        print("🛑 STT worker thread stopped")
    
    def add_file_for_transcription(self, file_path: str):
        """
        Add an audio file to the transcription queue.
        
        Args:
            file_path: Path to the audio file to transcribe
        """
        if not self.is_enabled:
            print(f"⚠️  STT not enabled, skipping: {file_path}")
            return
        
        if not os.path.exists(file_path):
            print(f"❌ Audio file not found: {file_path}")
            return
        
        try:
            self.transcription_queue.put_nowait(file_path)
            print(f"📥 Added to transcription queue: {os.path.basename(file_path)}")
        except queue.Full:
            print(f"⚠️  Transcription queue full, skipping: {os.path.basename(file_path)}")
    
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """
        Transcribe a single audio file synchronously.
        
        Args:
            file_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text, or None if transcription failed
        """
        if not self.is_enabled or not self.openai_client:
            print("❌ OpenAI STT not available")
            return None
        
        if not os.path.exists(file_path):
            print(f"❌ Audio file not found: {file_path}")
            return None
        
        try:
            print(f"🎤 Transcribing: {os.path.basename(file_path)}")
            
            with open(file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="text",
                    language = "en",
                    prompt=STTPrompt,
                )
            
            result = transcription.strip() if transcription else ""
            return result
            
        except Exception as e:
            print(f"❌ OpenAI transcription failed for {os.path.basename(file_path)}: {str(e)}")
            return None
    
    def _transcription_worker(self):
        """
        Background worker that processes the transcription queue.
        """
        print("🎯 STT transcription worker started")
        
        while self.is_running:
            try:
                # Get file from queue with timeout
                file_path = self.transcription_queue.get(timeout=1.0)
                
                # Transcribe the file
                transcription = self.transcribe_file(file_path)
                
                if transcription:
                    # Print transcription to terminal with formatting
                    print(f"\n🎙️  TRANSCRIPTION: {transcription}")
                    print("─" * 60)
                    print()  # Extra line for readability
                    
                    # Call callback if provided
                    if self.on_transcription_complete:
                        self.on_transcription_complete(transcription)
                else:
                    print(f"\n🔇 No speech detected in: {os.path.basename(file_path)}")
                    print("─" * 40)
                    print()
                
                # Mark task as done
                self.transcription_queue.task_done()
                
            except queue.Empty:
                # No files to process, continue waiting
                continue
            except Exception as e:
                print(f"❌ Error in transcription worker: {str(e)}")
    
    def get_queue_size(self) -> int:
        """
        Get the current size of the transcription queue.
        
        Returns:
            Number of files waiting for transcription
        """
        return self.transcription_queue.qsize()
    
    def is_queue_empty(self) -> bool:
        """
        Check if the transcription queue is empty.
        
        Returns:
            True if queue is empty, False otherwise
        """
        return self.transcription_queue.empty()
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """
        Wait for all queued transcriptions to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None for indefinite)
        """
        try:
            self.transcription_queue.join()
        except KeyboardInterrupt:
            print("\n🛑 Transcription wait interrupted")
    
    def get_stats(self) -> dict:
        """
        Get STT statistics.
        
        Returns:
            Dictionary with STT status and statistics
        """
        return {
            'enabled': self.is_enabled,
            'model': self.model,
            'worker_running': self.is_running,
            'queue_size': self.get_queue_size(),
            'client_available': self.openai_client is not None
        }


if __name__ == '__main__':
    # Test the STT transcriber
    print("🧪 Testing STT Transcriber")
    
    def on_transcription_complete(text):
        print(f"📝 Transcription callback: {text}")
    
    # Create STT instance
    stt = STTTranscriber(on_transcription_complete=on_transcription_complete)
    
    # Show stats
    stats = stt.get_stats()
    print(f"\n📊 STT Stats: {stats}")
    
    # Stop worker
    stt.stop_worker() 