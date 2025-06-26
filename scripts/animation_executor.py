#!/usr/bin/env python3

import subprocess
import time
from typing import List, Optional
import threading
from queue import Queue, Empty
from prompt import DiscoNames, DiscoKeys, DiscoTimes 

class NAOAnimationExecutor:
    """
    Tool for executing animations on NAO robot via SSH and qicli commands.
    
    Based on NAO robot animation capabilities referenced from:
    - https://github.com/berlonics/nao_description - NAO robot description
    - https://github.com/longnguyen1997/nao_animations - NAO animations research
    """
    
    def __init__(self, nao_ip: str = "nao.local", username: str = "nao", timeout: int = 10):
        """
        Initialize NAO animation executor.
        
        Args:
            nao_ip: IP address or hostname of the NAO robot
            username: SSH username (default: nao)
            timeout: SSH connection timeout in seconds
        """
        self.nao_ip = nao_ip
        self.username = username
        self.timeout = timeout
        self.is_executing = False
        self.execution_queue = Queue()
        self.worker_thread = None
        self.is_running = False
    
        
        print(f"🤖 NAO Animation Executor initialized for {username}@{nao_ip}")
    
    def test_connection(self) -> bool:
        """
        Test SSH connection to NAO robot.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            cmd = [
                "ssh", 
                f"{self.username}@{self.nao_ip}",
                "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=no",
                "echo 'Connection test successful'"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                print(f"✅ SSH connection to {self.username}@{self.nao_ip} successful")
                return True
            else:
                print(f"❌ SSH connection failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ SSH connection timeout to {self.nao_ip}")
            return False
        except Exception as e:
            print(f"❌ SSH connection error: {e}")
            return False
    
    def execute_single_animation(self, action: str,) -> bool:
        """
        Execute a single animation on NAO robot.
        
        Args:
            action: Animation action name or direct path
            animation_path: Optional specific animation path to execute
            
        Returns:
            True if animation executed successfully, False otherwise
        """
        print("/////////////////////////////////////")
        print(action)
        try:
            # Determine the command to use
            if action=="Crouch":
                qicli_command = f"qicli call ALRobotPosture.goToPosture Crouch 0.5"
                print(f"🎭 Executing animation path: {action}")
            elif action=="Stand":
                qicli_command = f"qicli call ALRobotPosture.goToPosture Stand 0.5"
                print(f"🎭 Executing animation path: {action}")
            elif action=="SingSong":
                qicli_command = f"qicli call ALAudioPlayer.playFile /home/nao/music/SingSongDoRiMe.mp3 1.0 0.0"
                print(f"🎭 Executing animation path: {action}")
            elif action=="DiscoDance":
                qicli_command = f"python /home/nao/code/NaoDanceDisco.py "
                print(f"🎭 Executing animation path: {action}")
            elif action.startswith("animations/"):
                # Direct animation path provided as action
                qicli_command = f"qicli call ALAnimationPlayer.run {action}"
                print(f"🎭 Executing animation path: {action}")
            else:
                # Try as tag if no mapping found
                qicli_command = f"qicli call ALAnimationPlayer.runTag {action}"
                print(f"🎭 Executing animation tag: {action}")
            
            # Build SSH command to execute qicli animation
            ssh_cmd = [
                "ssh",
                f"{self.username}@{self.nao_ip}",
                "-o", "ConnectTimeout=5", 
                "-o", "StrictHostKeyChecking=no",
                qicli_command
            ]
            
            # Execute SSH command
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Animation '{action}' executed successfully")
                return True
            else:
                print(f"❌ Animation '{action}' failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Animation '{action}' timed out")
            return False
        except Exception as e:
            print(f"❌ Animation '{action}' error: {e}")
            return False
    
    def execute_animation_sequence(self, actions: List[str], delay_between: float = 1.0) -> bool:
        """
        Execute a sequence of animations with delays.
        
        Args:
            actions: List of animation action names
            delay_between: Delay between animations in seconds
            
        Returns:
            True if all animations executed successfully, False otherwise
        """
        if not actions:
            print("⚠️  No animation actions provided")
            return False
        
        print(f"🎭 Starting animation sequence: {actions}")
        success_count = 0
        
        for i, action in enumerate(actions):
            success = self.execute_single_animation(action)
            if success:
                success_count += 1
            
            # Add delay between animations (except after the last one)
            if i < len(actions) - 1:
                time.sleep(delay_between)
        
        print(f"🎭 Animation sequence completed: {success_count}/{len(actions)} successful")
        return success_count == len(actions)
    
    def add_animation_request(self, actions: List[str], delay_between: float = 1.0):
        """
        Add animation request to the execution queue.
        
        Args:
            actions: List of animation action names
            delay_between: Delay between animations in seconds
        """
        if not self.is_running:
            self.start_worker()
        
        request = {
            'actions': actions,
            'delay_between': delay_between,
            'timestamp': time.time()
        }
        
        try:
            self.execution_queue.put(request, block=False)
            print(f"📝 Added animation request to queue: {actions}")
        except:
            print(f"⚠️  Animation queue is full, dropping request: {actions}")
    
    def start_worker(self):
        """Start the background worker thread for processing animation requests."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("🚀 NAO Animation worker thread started")
    
    def stop_worker(self):
        """Stop the background worker thread."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        print("🛑 NAO Animation worker thread stopped")
    
    def _worker_loop(self):
        """Background worker loop for processing animation requests."""
        print("🔄 NAO Animation worker loop started")
        
        while self.is_running:
            try:
                # Get request from queue with timeout
                request = self.execution_queue.get(timeout=1.0)
                self._process_request(request)
                self.execution_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Error in animation worker loop: {str(e)}")
        
        print("🔄 NAO Animation worker loop stopped")
    
    def get_current_posture(self) -> Optional[str]:
        """
        Get the current posture of the NAO robot.
        
        Returns:
            Current posture name as string, or None if failed
        """
        try:
            qicli_command = "qicli call ALRobotPosture.getPosture"
            
            ssh_cmd = [
                "ssh",
                f"{self.username}@{self.nao_ip}",
                "-o", "ConnectTimeout=5", 
                "-o", "StrictHostKeyChecking=no",
                qicli_command
            ]
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                posture = result.stdout.strip()
                print(f"📍 Current posture: {posture}")
                return posture
            else:
                print(f"❌ Failed to get current posture: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Get posture command timed out")
            return None
        except Exception as e:
            print(f"❌ Get posture error: {e}")
            return None
    
    def restore_posture(self, posture: str, speed: float = 0.5) -> bool:
        """
        Restore NAO robot to a specific posture.
        
        Args:
            posture: Target posture name
            speed: Speed of posture transition (0.0 to 1.0)
            
        Returns:
            True if posture restored successfully, False otherwise
        """
        try:
            qicli_command = f"qicli call ALRobotPosture.goToPosture {posture} {speed}"
            
            ssh_cmd = [
                "ssh",
                f"{self.username}@{self.nao_ip}",
                "-o", "ConnectTimeout=5", 
                "-o", "StrictHostKeyChecking=no",
                qicli_command
            ]
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                print(f"🔄 Restored to posture: {posture}")
                return True
            else:
                print(f"❌ Failed to restore posture '{posture}': {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Restore posture command timed out")
            return False
        except Exception as e:
            print(f"❌ Restore posture error: {e}")
            return False

    def _process_request(self, request: dict):
        """
        Process a single animation request.
        
        Args:
            request: Animation request dictionary
        """
        original_posture = None
        try:
            actions = request['actions']
            delay_between = request.get('delay_between', 1.0)
            
            print(f"🤖 Processing animation request: {actions}")
            self.is_executing = True
            
            # Get current posture before executing animations
            original_posture = self.get_current_posture()
            if original_posture is None:
                print("⚠️  Could not get current posture, proceeding without posture restoration")
            
            # Execute the animation sequence
            success = self.execute_animation_sequence(actions, delay_between)
            
            if success:
                print(f"✅ Animation request completed successfully")
            else:
                print(f"⚠️  Animation request completed with some failures")
                
        except Exception as e:
            print(f"❌ Animation request processing failed: {str(e)}")
        finally:
            # Restore original posture if we successfully got it
            if original_posture is not None:
                print(f"🔄 Restoring original posture: {original_posture}")
                restore_success = self.restore_posture(original_posture)
                if not restore_success:
                    print(f"⚠️  Failed to restore original posture: {original_posture}")
            
            self.is_executing = False
    
    def is_busy(self) -> bool:
        """
        Check if the executor is currently busy.
        
        Returns:
            True if currently executing animations
        """
        return self.is_executing or not self.execution_queue.empty()
    
    def get_queue_size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of animation requests in queue
        """
        return self.execution_queue.qsize()

    def execute_direct_animation(self, action_key: str) -> bool:
        """
        Execute a specific animation directly by its path.
        
        Args:
            action_key: The action key (for logging)
            animation_path: Direct path to the animation
            
        Returns:
            True if animation executed successfully, False otherwise
        """
        original_posture = None
        try:
            # Get current posture before executing animation
            original_posture = self.get_current_posture()
            if original_posture is None:
                print("⚠️  Could not get current posture, proceeding without posture restoration")
            
            # Execute the animation
            success = self.execute_single_animation(action_key)
            
            return success
            
        except Exception as e:
            print(f"❌ Direct animation execution failed: {str(e)}")
            return False
        finally:
            # Restore original posture if we successfully got it
            if original_posture is not None and action_key!="Crouch" and action_key!="Stand":
                print(f"🔄 Restoring original posture: {original_posture}")
                restore_success = self.restore_posture(original_posture)
                if not restore_success:
                    print(f"⚠️  Failed to restore original posture: {original_posture}")
            else:
                print("ℹ️  No posture restoration needed")
    

    def update_nao_ip(self, new_ip: str):
        """
        Update NAO robot IP address.
        
        Args:
            new_ip: New IP address or hostname
        """
        self.nao_ip = new_ip
        print(f"🔄 Updated NAO IP to: {new_ip}")


# Convenience function for quick animation execution
def execute_nao_animations(actions: List[str], nao_ip: str = "nao.local", delay_between: float = 1.0) -> bool:
    """
    Convenience function to execute NAO animations.
    
    Args:
        actions: List of animation action names
        nao_ip: NAO robot IP address
        delay_between: Delay between animations in seconds
        
    Returns:
        True if all animations executed successfully
    """
    executor = NAOAnimationExecutor(nao_ip=nao_ip)
    return executor.execute_animation_sequence(actions, delay_between)


if __name__ == '__main__':
    # Test the animation executor
    executor = NAOAnimationExecutor()
    
    # Test connection
    if executor.test_connection():
        # Test single animation
        executor.execute_single_animation("joy")
        
        # Test animation sequence
        test_actions = ["confirmation", "joy", "user"]
        executor.execute_animation_sequence(test_actions, delay_between=2.0)
    else:
        print("❌ Cannot connect to NAO robot, skipping animation tests")
        
