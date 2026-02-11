"""
Conversation Manager Module
Handles conversation history storage and session management.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class ConversationManager:
    """
    Manages conversation sessions and history.
    Stores Q&A turns with timestamps and context for continuity.
    """
    
    def __init__(self, storage_path: str = "./data/conversations"):
        """
        Initialize conversation manager.
        
        Args:
            storage_path: Directory to store conversation sessions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache: {session_id: session_data}
        self.sessions = {}
        
        # Session metadata
        self.session_metadata = {}
    
    def create_session(self, user_id: str = None, session_id: str = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            user_id: Optional user identifier
            session_id: Optional session ID (generates new UUID if None)
            
        Returns:
            Session ID (UUID)
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'turns': [],
            'created_at': datetime.now().isoformat(),
            'user_id': user_id,
            'last_updated': datetime.now().isoformat()
        }
        
        return session_id
    
    def add_turn(self, 
                 session_id: str, 
                 question: str, 
                 answer: str,
                 retrieved_context: List[Dict] = None,
                 metadata: Dict = None) -> None:
        """
        Add a conversation turn to a session.
        
        Args:
            session_id: Session identifier
            question: User's question
            answer: System's answer
            retrieved_context: Retrieved chunks used for context
            metadata: Additional metadata (top_k, model, etc.)
        """
        if session_id not in self.sessions:
            # Load from disk if exists, otherwise create
            if not self._load_session(session_id):
                self.create_session(session_id=session_id)
        
        turn = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'turn_number': len(self.sessions[session_id]['turns']) + 1,
            'retrieved_context': retrieved_context or [],
            'metadata': metadata or {}
        }
        
        self.sessions[session_id]['turns'].append(turn)
        self.sessions[session_id]['last_updated'] = datetime.now().isoformat()
        
        # Auto-save after each turn
        self.save_session(session_id)
    
    def get_history(self, session_id: str, last_n: int = None) -> List[Dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            last_n: Number of recent turns to return (None = all)
            
        Returns:
            List of conversation turns
        """
        if session_id not in self.sessions:
            if not self._load_session(session_id):
                return []
        
        turns = self.sessions[session_id]['turns']
        
        if last_n is not None:
            return turns[-last_n:]
        
        return turns
    
    def get_recent_context(self, session_id: str, last_n: int = 3) -> str:
        """
        Get recent conversation context as formatted string.
        
        Args:
            session_id: Session identifier
            last_n: Number of recent turns to include
            
        Returns:
            Formatted conversation context
        """
        history = self.get_history(session_id, last_n)
        
        if not history:
            return ""
        
        context_parts = []
        for turn in history:
            context_parts.append(f"Q: {turn['question']}")
            context_parts.append(f"A: {turn['answer'][:200]}...")  # Truncate long answers
        
        return "\n".join(context_parts)
    
    def get_last_question(self, session_id: str) -> Optional[str]:
        """
        Get the last question from a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Last question or None
        """
        history = self.get_history(session_id, last_n=1)
        
        if history:
            return history[0]['question']
        
        return None
    
    def save_session(self, session_id: str) -> bool:
        """
        Save session to disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        if session_id not in self.sessions:
            return False
        
        try:
            session_file = self.storage_path / f"{session_id}.pkl"
            with open(session_file, 'wb') as f:
                pickle.dump(self.sessions[session_id], f)
            return True
        except Exception as e:
            print(f"⚠️ Error saving session {session_id}: {e}")
            return False
    
    def _load_session(self, session_id: str) -> bool:
        """
        Load session from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if loaded successfully
        """
        session_file = self.storage_path / f"{session_id}.pkl"
        
        if not session_file.exists():
            return False
        
        try:
            with open(session_file, 'rb') as f:
                self.sessions[session_id] = pickle.load(f)
            return True
        except Exception as e:
            print(f"⚠️ Error loading session {session_id}: {e}")
            return False
    
    def list_sessions(self, user_id: str = None) -> List[Dict]:
        """
        List all sessions (optionally filtered by user).
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            List of session summary dicts
        """
        sessions = []
        
        # Load all session files
        for session_file in self.storage_path.glob("*.pkl"):
            session_id = session_file.stem
            
            if session_id not in self.sessions:
                self._load_session(session_id)
            
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Filter by user_id if provided
                if user_id and session.get('user_id') != user_id:
                    continue
                
                sessions.append({
                    'session_id': session_id,
                    'created_at': session.get('created_at'),
                    'last_updated': session.get('last_updated'),
                    'turn_count': len(session.get('turns', [])),
                    'user_id': session.get('user_id')
                })
        
        # Sort by last_updated (most recent first)
        sessions.sort(key=lambda x: x['last_updated'], reverse=True)
        
        return sessions
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session from memory and disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        # Remove from memory
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Remove from disk
        session_file = self.storage_path / f"{session_id}.pkl"
        if session_file.exists():
            try:
                session_file.unlink()
                return True
            except Exception as e:
                print(f"⚠️ Error deleting session {session_id}: {e}")
                return False
        
        return True
    
    def get_session_stats(self, session_id: str) -> Dict:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        if session_id not in self.sessions:
            if not self._load_session(session_id):
                return {}
        
        session = self.sessions[session_id]
        turns = session.get('turns', [])
        
        return {
            'session_id': session_id,
            'turn_count': len(turns),
            'created_at': session.get('created_at'),
            'last_updated': session.get('last_updated'),
            'avg_question_length': sum(len(t['question']) for t in turns) / len(turns) if turns else 0,
            'avg_answer_length': sum(len(t['answer']) for t in turns) / len(turns) if turns else 0,
            'user_id': session.get('user_id')
        }
