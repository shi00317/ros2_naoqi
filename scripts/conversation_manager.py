#!/usr/bin/env python3

"""
NAO Conversation Manager Utility

This script provides utilities for managing persistent conversation states.
It can be used to list, load, save, and delete conversation sessions.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional

def list_conversations(save_directory: str = "/tmp/nao_conversations") -> List[Dict]:
    """
    List all saved conversation sessions.
    
    Args:
        save_directory: Directory containing saved conversations
        
    Returns:
        List of conversation metadata
    """
    conversations = []
    save_path = Path(save_directory)
    
    if not save_path.exists():
        print(f"⚠️  Save directory does not exist: {save_directory}")
        return conversations
    
    try:
        for file_path in save_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                conversations.append({
                    'session_id': state_data.get('session_id'),
                    'user_id': state_data.get('user_id'),
                    'message_count': len(state_data.get('messages', [])),
                    'last_activity': state_data.get('last_activity'),
                    'saved_at': state_data.get('saved_at'),
                    'model': state_data.get('model'),
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size
                })
            except Exception as e:
                print(f"⚠️  Failed to read conversation file {file_path}: {e}")
                
    except Exception as e:
        print(f"❌ Failed to list conversations: {e}")
    
    return sorted(conversations, key=lambda x: x.get('last_activity', 0), reverse=True)

def show_conversation_details(session_id: str, save_directory: str = "/tmp/nao_conversations"):
    """
    Show detailed information about a specific conversation.
    
    Args:
        session_id: Session identifier
        save_directory: Directory containing saved conversations
    """
    file_path = Path(save_directory) / f"{session_id}.json"
    
    if not file_path.exists():
        print(f"❌ Conversation not found: {session_id}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        print(f"\n📋 Conversation Details: {session_id}")
        print("=" * 60)
        print(f"Session ID: {state_data.get('session_id')}")
        print(f"User ID: {state_data.get('user_id')}")
        print(f"Model: {state_data.get('model')}")
        print(f"Messages: {len(state_data.get('messages', []))}")
        print(f"Last Activity: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state_data.get('last_activity', 0)))}")
        print(f"Saved At: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state_data.get('saved_at', 0)))}")
        print(f"File Size: {file_path.stat().st_size} bytes")
        print(f"File Path: {file_path}")
        
        messages = state_data.get('messages', [])
        if messages:
            print(f"\n💬 Conversation History ({len(messages)} messages):")
            print("-" * 60)
            for i, msg in enumerate(messages, 1):
                role_icon = "👤" if msg['role'] == 'user' else "🤖"
                timestamp = time.strftime("%H:%M:%S", time.localtime(msg.get('timestamp', 0)))
                print(f"{i:2d}. {role_icon} [{timestamp}] {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                if msg.get('error'):
                    print(f"     ⚠️  Error occurred during this message")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Failed to read conversation details: {e}")

def delete_conversation(session_id: str, save_directory: str = "/tmp/nao_conversations") -> bool:
    """
    Delete a saved conversation.
    
    Args:
        session_id: Session identifier to delete
        save_directory: Directory containing saved conversations
        
    Returns:
        True if deleted successfully
    """
    file_path = Path(save_directory) / f"{session_id}.json"
    
    if not file_path.exists():
        print(f"❌ Conversation not found: {session_id}")
        return False
    
    try:
        file_path.unlink()
        print(f"🗑️  Deleted conversation: {session_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete conversation {session_id}: {e}")
        return False

def cleanup_old_conversations(save_directory: str = "/tmp/nao_conversations", 
                            days_old: int = 30) -> int:
    """
    Clean up conversations older than specified days.
    
    Args:
        save_directory: Directory containing saved conversations
        days_old: Delete conversations older than this many days
        
    Returns:
        Number of conversations deleted
    """
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    deleted_count = 0
    
    conversations = list_conversations(save_directory)
    
    for conv in conversations:
        last_activity = conv.get('last_activity', 0)
        if last_activity < cutoff_time:
            if delete_conversation(conv['session_id'], save_directory):
                deleted_count += 1
    
    print(f"🧹 Cleaned up {deleted_count} conversations older than {days_old} days")
    return deleted_count

def export_conversation_to_text(session_id: str, 
                              save_directory: str = "/tmp/nao_conversations",
                              output_file: Optional[str] = None) -> Optional[str]:
    """
    Export a conversation to a readable text file.
    
    Args:
        session_id: Session identifier
        save_directory: Directory containing saved conversations
        output_file: Output file path (auto-generated if None)
        
    Returns:
        Path to exported file or None if failed
    """
    file_path = Path(save_directory) / f"{session_id}.json"
    
    if not file_path.exists():
        print(f"❌ Conversation not found: {session_id}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        if output_file is None:
            output_file = f"{session_id}_conversation.txt"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"NAO Robot Conversation Export\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"User ID: {state_data.get('user_id')}\n")
            f.write(f"Model: {state_data.get('model')}\n")
            f.write(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Last Activity: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state_data.get('last_activity', 0)))}\n")
            f.write("=" * 80 + "\n\n")
            
            messages = state_data.get('messages', [])
            for i, msg in enumerate(messages, 1):
                role = "USER" if msg['role'] == 'user' else "NAO"
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.get('timestamp', 0)))
                f.write(f"[{i:03d}] {role} ({timestamp}):\n")
                f.write(f"{msg['content']}\n\n")
                if msg.get('error'):
                    f.write("    ⚠️  Error occurred during this message\n\n")
        
        print(f"📄 Exported conversation to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Failed to export conversation: {e}")
        return None

def main():
    """Main CLI interface for conversation management."""
    if len(sys.argv) < 2:
        print("NAO Conversation Manager")
        print("Usage:")
        print("  python conversation_manager.py list [save_directory]")
        print("  python conversation_manager.py show <session_id> [save_directory]")
        print("  python conversation_manager.py delete <session_id> [save_directory]")
        print("  python conversation_manager.py cleanup [days_old] [save_directory]")
        print("  python conversation_manager.py export <session_id> [output_file] [save_directory]")
        print("")
        print("Examples:")
        print("  python conversation_manager.py list")
        print("  python conversation_manager.py show nao_session")
        print("  python conversation_manager.py delete old_session")
        print("  python conversation_manager.py cleanup 7")
        print("  python conversation_manager.py export nao_session conversation.txt")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    save_directory = "/tmp/nao_conversations"
    
    if command == "list":
        if len(sys.argv) > 2:
            save_directory = sys.argv[2]
        
        conversations = list_conversations(save_directory)
        
        if not conversations:
            print("📭 No saved conversations found")
            return
        
        print(f"\n💬 Saved Conversations ({len(conversations)} total):")
        print("=" * 100)
        print(f"{'Session ID':<25} {'Messages':<8} {'Last Activity':<20} {'Size':<8} {'Model'}")
        print("-" * 100)
        
        for conv in conversations:
            last_activity = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(conv.get('last_activity', 0)))
            size_kb = conv.get('file_size', 0) / 1024
            print(f"{conv['session_id']:<25} {conv['message_count']:<8} {last_activity:<20} {size_kb:<7.1f}K {conv.get('model', 'N/A')}")
        
        print("=" * 100)
        print(f"Total conversations: {len(conversations)}")
        
    elif command == "show":
        if len(sys.argv) < 3:
            print("❌ Session ID required")
            sys.exit(1)
        
        session_id = sys.argv[2]
        if len(sys.argv) > 3:
            save_directory = sys.argv[3]
        
        show_conversation_details(session_id, save_directory)
        
    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ Session ID required")
            sys.exit(1)
        
        session_id = sys.argv[2]
        if len(sys.argv) > 3:
            save_directory = sys.argv[3]
        
        delete_conversation(session_id, save_directory)
        
    elif command == "cleanup":
        days_old = 30
        if len(sys.argv) > 2:
            try:
                days_old = int(sys.argv[2])
            except ValueError:
                print("❌ Invalid days_old value")
                sys.exit(1)
        
        if len(sys.argv) > 3:
            save_directory = sys.argv[3]
        
        cleanup_old_conversations(save_directory, days_old)
        
    elif command == "export":
        if len(sys.argv) < 3:
            print("❌ Session ID required")
            sys.exit(1)
        
        session_id = sys.argv[2]
        output_file = None
        
        if len(sys.argv) > 3:
            output_file = sys.argv[3]
        
        if len(sys.argv) > 4:
            save_directory = sys.argv[4]
        
        export_conversation_to_text(session_id, save_directory, output_file)
        
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main() 