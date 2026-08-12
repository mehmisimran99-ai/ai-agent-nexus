# AI Agent Nexus - Conversation Memory
"""
Conversation memory with thread persistence, summarization, and context management
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json
import os

from app.agents.state import AgentMessage


class ConversationThread:
    """Single conversation thread"""
    
    def __init__(self, thread_id: str, title: str, metadata: Dict[str, Any] = None):
        self.id = thread_id
        self.title = title
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.messages: List[AgentMessage] = []
        self.metadata = metadata or {}
        self.summary: Optional[str] = None
    
    def add_message(self, message: AgentMessage):
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
            "metadata": self.metadata,
        }


class ConversationMemory:
    """Conversation memory manager"""
    
    def __init__(self, storage_path: str = "./data/conversations"):
        self.storage_path = storage_path
        self.threads: Dict[str, ConversationThread] = {}
        os.makedirs(storage_path, exist_ok=True)
        self._load_threads()
    
    def _load_threads(self):
        """Load threads from disk"""
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                thread_id = filename[:-5]
                path = os.path.join(self.storage_path, filename)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    thread = ConversationThread(
                        thread_id=data["id"],
                        title=data["title"],
                        metadata=data.get("metadata", {}),
                    )
                    thread.created_at = datetime.fromisoformat(data["created_at"])
                    thread.updated_at = datetime.fromisoformat(data["updated_at"])
                    thread.summary = data.get("summary")
                    # Messages would need deserialization
                    self.threads[thread_id] = thread
                except Exception:
                    pass
    
    def _save_thread(self, thread: ConversationThread):
        """Save thread to disk"""
        path = os.path.join(self.storage_path, f"{thread.id}.json")
        data = thread.to_dict()
        data["summary"] = thread.summary
        # Note: messages not serialized in this simplified version
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    async def create_thread(self, title: str, metadata: Dict[str, Any] = None) -> ConversationThread:
        """Create a new conversation thread"""
        thread_id = str(uuid.uuid4())
        thread = ConversationThread(thread_id, title, metadata)
        self.threads[thread_id] = thread
        self._save_thread(thread)
        return thread
    
    async def list_threads(self, limit: int = 50, offset: int = 0) -> List[ConversationThread]:
        """List conversation threads"""
        sorted_threads = sorted(
            self.threads.values(),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        return sorted_threads[offset:offset + limit]
    
    async def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        """Get a thread by ID"""
        return self.threads.get(thread_id)
    
    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread"""
        if thread_id in self.threads:
            del self.threads[thread_id]
            path = os.path.join(self.storage_path, f"{thread_id}.json")
            if os.path.exists(path):
                os.remove(path)
            return True
        return False
    
    async def add_message(self, thread_id: str, message: AgentMessage) -> bool:
        """Add a message to a thread"""
        thread = self.threads.get(thread_id)
        if thread:
            thread.add_message(message)
            self._save_thread(thread)
            return True
        return False
    
    async def get_messages(self, thread_id: str, limit: int = 100) -> List[AgentMessage]:
        """Get messages from a thread"""
        thread = self.threads.get(thread_id)
        if thread:
            return thread.messages[-limit:]
        return []
    
    async def summarize_thread(self, thread_id: str) -> Optional[str]:
        """Generate a summary of the thread"""
        thread = self.threads.get(thread_id)
        if not thread or not thread.messages:
            return None
        
        # In production: use LLM to generate summary
        # For now: simple heuristic summary
        user_messages = [m for m in thread.messages if m.role == "user"]
        topics = [m.content[:50] for m in user_messages[:5]]
        
        summary = f"Conversation with {len(thread.messages)} messages. Topics: {', '.join(topics)}"
        thread.summary = summary
        self._save_thread(thread)
        
        return summary
    
    async def get_context_window(self, thread_id: str, max_messages: int = 20) -> List[AgentMessage]:
        """Get recent messages for context window"""
        thread = self.threads.get(thread_id)
        if thread:
            return thread.messages[-max_messages:]
        return []