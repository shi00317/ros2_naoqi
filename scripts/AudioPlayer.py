#!/usr/bin/env python3

import os
import subprocess
import time
from typing import Optional
from pathlib import Path


class NaoAudioPlayer:
    """
    Audio player specifically designed for NAO robot.
    Uses SCP to transfer files and qicli to play audio via ALAudioPlayer.
    """
    
    def __init__(self, 
                 nao_ip: str = os.getenv("NAO_IP", "127.0.0.1"),
                 nao_user: str = "nao",
                 nao_password: str = "nao",
                 nao_audio_dir: str = "/tmp/audio"):
        """
        Initialize the NAO audio player.
        
        Args:
            nao_ip: IP address of the NAO robot
            nao_user: Username for NAO robot (default: nao)
            nao_password: Password for NAO robot (default: nao)
            nao_audio_dir: Directory on NAO to store audio files
        """
        self.nao_ip = nao_ip
        self.nao_user = nao_user
        self.nao_password = nao_password
        self.nao_audio_dir = nao_audio_dir
        
        self.stats = {
            'playback_attempts': 0,
            'playback_successes': 0,
            'playback_failures': 0,
            'scp_transfers': 0,
            'scp_failures': 0,
            'qicli_calls': 0,
            'qicli_failures': 0,
            'last_file_played': None
        }
        
        print(f"🔊 NAO Audio Player initialized")
        print(f"   🤖 NAO IP: {self.nao_ip}")
        print(f"   👤 NAO User: {self.nao_user}")
        print(f"   📁 NAO Audio Dir: {self.nao_audio_dir}")
    
    def play_audio_file(self, audio_file_path: str, timeout: int = 30, volume: float = 1.0, pan: float = 0.0) -> bool:
        """
        Play an audio file on NAO robot using SCP transfer and qicli ALAudioPlayer.
        
        Args:
            audio_file_path: Path to the local audio file to play
            timeout: Maximum time to wait for operations (seconds)
            volume: Volume level (0.0 to 1.0)
            pan: Pan level (-1.0 to 1.0, where -1.0 is left, 1.0 is right)
            
        Returns:
            True if playback was successful, False otherwise
        """
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            self.stats['playback_failures'] += 1
            return False
        
        self.stats['playback_attempts'] += 1
        print(f"🔊 Playing audio on NAO: {os.path.basename(audio_file_path)}")
        
        try:
            # Step 1: Transfer file to NAO robot
            if not self._transfer_file_to_nao(audio_file_path, timeout):
                self.stats['playback_failures'] += 1
                return False
            
            # Step 2: Play file on NAO using qicli
            remote_file_path = f"{self.nao_audio_dir}/{os.path.basename(audio_file_path)}"
            if self._play_file_on_nao(remote_file_path, volume, pan, timeout):
                self.stats['playback_successes'] += 1
                self.stats['last_file_played'] = os.path.basename(audio_file_path)
                print(f"✅ Audio played successfully on NAO")
                return True
            else:
                self.stats['playback_failures'] += 1
                return False
                
        except Exception as e:
            self.stats['playback_failures'] += 1
            print(f"❌ Error playing audio on NAO: {str(e)}")
            return False
    
    def _transfer_file_to_nao(self, local_file_path: str, timeout: int) -> bool:
        """
        Transfer an audio file to NAO robot using SCP.
        
        Args:
            local_file_path: Path to local audio file
            timeout: Timeout in seconds
            
        Returns:
            True if transfer was successful, False otherwise
        """
        try:
            self.stats['scp_transfers'] += 1
            
            # Create remote directory if it doesn't exist
            mkdir_cmd = [
                'sshpass', '-p', self.nao_password,
                'ssh', f'{self.nao_user}@{self.nao_ip}',
                f'mkdir -p {self.nao_audio_dir}'
            ]
            
            print(f"📁 Creating directory on NAO: {self.nao_audio_dir}")
            subprocess.run(mkdir_cmd, capture_output=True, timeout=timeout, check=False)
            
            # Transfer file using SCP
            filename = os.path.basename(local_file_path)
            remote_path = f"{self.nao_user}@{self.nao_ip}:{self.nao_audio_dir}/{filename}"
            
            scp_cmd = [
                'sshpass', '-p', self.nao_password,
                'scp', local_file_path, remote_path
            ]
            
            print(f"📤 Transferring {filename} to NAO...")
            result = subprocess.run(scp_cmd, capture_output=True, timeout=timeout, check=False)
            
            if result.returncode == 0:
                print(f"✅ File transferred successfully to NAO")
                return True
            else:
                print(f"❌ SCP transfer failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr.decode()}")
                self.stats['scp_failures'] += 1
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  SCP transfer timed out")
            self.stats['scp_failures'] += 1
            return False
        except FileNotFoundError:
            print(f"⚠️  sshpass or scp not found. Install with: sudo apt install sshpass openssh-client")
            self.stats['scp_failures'] += 1
            return False
        except Exception as e:
            print(f"⚠️  SCP transfer error: {str(e)}")
            self.stats['scp_failures'] += 1
            return False
    
    def _play_file_on_nao(self, remote_file_path: str, volume: float, pan: float, timeout: int) -> bool:
        """
        Play an audio file on NAO robot using qicli ALAudioPlayer.
        
        Args:
            remote_file_path: Path to audio file on NAO robot
            volume: Volume level (0.0 to 1.0)
            pan: Pan level (-1.0 to 1.0)
            timeout: Timeout in seconds
            
        Returns:
            True if playback was successful, False otherwise
        """
        try:
            self.stats['qicli_calls'] += 1
            
            # Choose qicli command based on parameters
            if volume != 1.0 or pan != 0.0:
                # Use playFile with volume and pan
                qicli_cmd = [
                    'sshpass', '-p', self.nao_password,
                    'ssh', f'{self.nao_user}@{self.nao_ip}',
                    f'qicli call ALAudioPlayer.playFile "{remote_file_path}" {volume} {pan}'
                ]
                print(f"🎵 Playing on NAO with volume={volume}, pan={pan}")
            else:
                # Use simple playFile
                qicli_cmd = [
                    'sshpass', '-p', self.nao_password,
                    'ssh', f'{self.nao_user}@{self.nao_ip}',
                    f'qicli call ALAudioPlayer.playFile "{remote_file_path}"'
                ]
                print(f"🎵 Playing on NAO with default settings")
            
            result = subprocess.run(qicli_cmd, capture_output=True, timeout=timeout, check=False)
            
            if result.returncode == 0:
                print(f"✅ qicli playFile completed successfully")
                return True
            else:
                print(f"❌ qicli playFile failed with return code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr.decode()}")
                self.stats['qicli_failures'] += 1
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  qicli playFile timed out")
            self.stats['qicli_failures'] += 1
            return False
        except Exception as e:
            print(f"⚠️  qicli playFile error: {str(e)}")
            self.stats['qicli_failures'] += 1
            return False
    
    def test_nao_connection(self) -> bool:
        """
        Test the connection to NAO robot.
        
        Returns:
            True if connection is successful, False otherwise
        """
        print("🧪 Testing NAO robot connection...")
        
        try:
            # Test SSH connection to NAO
            test_cmd = [
                'sshpass', '-p', self.nao_password,
                'ssh', '-o', 'ConnectTimeout=10', f'{self.nao_user}@{self.nao_ip}',
                'echo "NAO connection test successful"'
            ]
            
            result = subprocess.run(test_cmd, capture_output=True, timeout=15, check=False)
            
            if result.returncode == 0:
                print(f"✅ NAO SSH connection: Success")
                
                # Test qicli availability
                qicli_test_cmd = [
                    'sshpass', '-p', self.nao_password,
                    'ssh', f'{self.nao_user}@{self.nao_ip}',
                    'qicli info ALAudioPlayer.playFile'
                ]
                
                qicli_result = subprocess.run(qicli_test_cmd, capture_output=True, timeout=10, check=False)
                
                if qicli_result.returncode == 0:
                    print(f"✅ qicli ALAudioPlayer: Available")
                    return True
                else:
                    print(f"❌ qicli ALAudioPlayer: Not available")
                    return False
            else:
                print(f"❌ NAO SSH connection: Failed")
                if result.stderr:
                    print(f"   Error: {result.stderr.decode()}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ NAO connection test: Timed out")
            return False
        except FileNotFoundError:
            print(f"❌ sshpass not found. Install with: sudo apt install sshpass")
            return False
        except Exception as e:
            print(f"❌ NAO connection test error: {str(e)}")
            return False
    
    def play_test_sound(self) -> bool:
        """
        Play a test sound on NAO robot to verify audio system is working.
        
        Returns:
            True if test sound played successfully
        """
        print("🔔 Playing test sound on NAO...")
        
        try:
            # Use NAO's built-in sound using ALTextToSpeech
            test_cmd = [
                'sshpass', '-p', self.nao_password,
                'ssh', f'{self.nao_user}@{self.nao_ip}',
                'qicli call ALTextToSpeech.say "Audio test successful"'
            ]
            
            result = subprocess.run(test_cmd, capture_output=True, timeout=15, check=False)
            
            if result.returncode == 0:
                print("✅ Test sound played successfully on NAO")
                return True
            else:
                print("❌ Unable to play test sound on NAO")
                if result.stderr:
                    print(f"   Error: {result.stderr.decode()}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Test sound timed out")
            return False
        except Exception as e:
            print(f"❌ Test sound error: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get NAO audio player statistics.
        
        Returns:
            Dictionary with playback statistics
        """
        return {
            'playback_attempts': self.stats['playback_attempts'],
            'playback_successes': self.stats['playback_successes'],
            'playback_failures': self.stats['playback_failures'],
            'scp_transfers': self.stats['scp_transfers'],
            'scp_failures': self.stats['scp_failures'],
            'qicli_calls': self.stats['qicli_calls'],
            'qicli_failures': self.stats['qicli_failures'],
            'last_file_played': self.stats['last_file_played'],
            'success_rate': (
                self.stats['playback_successes'] / max(1, self.stats['playback_attempts']) * 100
            )
        }
    
    def get_nao_config(self) -> dict:
        """
        Get NAO robot configuration.
        
        Returns:
            Dictionary with NAO configuration
        """
        return {
            'nao_ip': self.nao_ip,
            'nao_user': self.nao_user,
            'nao_audio_dir': self.nao_audio_dir
        }


if __name__ == '__main__':
    # Test the NAO audio player
    player = NaoAudioPlayer()
    
    # Test NAO connection
    connected = player.test_nao_connection()
    print(f"\n📊 NAO Connected: {connected}")
    
    # Test NAO audio system
    if connected:
        player.play_test_sound()
    
    # Show stats
    stats = player.get_stats()
    print(f"\n📊 NAO Audio Player Stats: {stats}")
    
    # Show config
    config = player.get_nao_config()
    print(f"\n⚙️  NAO Configuration: {config}") 