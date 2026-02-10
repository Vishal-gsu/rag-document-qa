"""
Query Rewriter Module
Rewrites user queries based on conversation context using LLM.
Resolves pronouns, implicit references, and contextual dependencies.
"""
from typing import Optional, List, Dict
from llm_manager import LLMManager


class QueryRewriter:
    """
    Rewrites user queries to be context-aware and standalone.
    Uses LLM to resolve pronouns and implicit references.
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        """
        Initialize query rewriter.
        
        Args:
            llm_manager: LLM manager for generating rewrites
        """
        self.llm_manager = llm_manager or LLMManager()
    
    def rewrite_with_context(self, 
                             current_query: str,
                             conversation_history: List[Dict],
                             enable_rewrite: bool = True) -> str:
        """
        Rewrite query based on conversation context.
        
        Args:
            current_query: User's current question
            conversation_history: Previous conversation turns
            enable_rewrite: Whether to enable LLM-based rewriting
            
        Returns:
            Rewritten query (standalone and context-aware)
        """
        # If no history or rewriting disabled, return original
        if not conversation_history or not enable_rewrite:
            return current_query
        
        # Check if query needs rewriting
        if not self._needs_rewriting(current_query):
            return current_query
        
        # Build conversation context
        context = self._build_context_prompt(current_query, conversation_history)
        
        # Use LLM to rewrite
        rewritten = self._llm_rewrite(context)
        
        # Fallback to original if rewriting fails
        if not rewritten or len(rewritten.strip()) == 0:
            return current_query
        
        return rewritten.strip()
    
    def _needs_rewriting(self, query: str) -> bool:
        """
        Check if query contains pronouns or context-dependent terms.
        
        Args:
            query: User's question
            
        Returns:
            True if rewriting is needed
        """
        query_lower = query.lower()
        
        # Pronouns that indicate context dependency
        pronouns = ['it', 'its', 'this', 'that', 'these', 'those', 'they', 'them', 'their']
        
        # Context-dependent phrases
        context_phrases = [
            'what about',
            'how about',
            'tell me more',
            'explain it',
            'explain that',
            'expand on',
            'also',
            'in addition',
            'furthermore',
            'similarly',
            'compared to',
            'versus',
            'vs'
        ]
        
        # Check for pronouns
        words = query_lower.split()
        if any(pronoun in words for pronoun in pronouns):
            return True
        
        # Check for context phrases
        if any(phrase in query_lower for phrase in context_phrases):
            return True
        
        # Check if query is very short (likely depends on context)
        if len(words) <= 3:
            return True
        
        return False
    
    def _build_context_prompt(self, 
                              current_query: str,
                              history: List[Dict],
                              max_turns: int = 3) -> str:
        """
        Build prompt for LLM rewriting.
        
        Args:
            current_query: User's current question
            history: Conversation history
            max_turns: Maximum turns to include
            
        Returns:
            Formatted prompt
        """
        # Get recent turns
        recent_history = history[-max_turns:] if len(history) > max_turns else history
        
        # Build conversation context
        context_parts = []
        for turn in recent_history:
            context_parts.append(f"User: {turn['question']}")
            # Truncate long answers
            answer_preview = turn['answer'][:300]
            if len(turn['answer']) > 300:
                answer_preview += "..."
            context_parts.append(f"Assistant: {answer_preview}")
        
        conversation_context = "\n".join(context_parts)
        
        # Build rewriting prompt
        prompt = f"""You are a query rewriting assistant. Your task is to rewrite the user's current question to be standalone and context-aware.

Conversation History:
{conversation_context}

Current User Question: {current_query}

Instructions:
1. Rewrite the current question to be completely standalone
2. Replace pronouns (it, this, that, they, etc.) with specific references from the conversation
3. Expand implicit references to be explicit
4. Keep the rewritten question concise and natural
5. If the question is already standalone, return it unchanged
6. Only output the rewritten question, nothing else

Rewritten Question:"""
        
        return prompt
    
    def _llm_rewrite(self, prompt: str) -> str:
        """
        Use LLM to rewrite query.
        
        Args:
            prompt: Rewriting prompt
            
        Returns:
            Rewritten query
        """
        try:
            # Use LLM to generate rewrite
            response = self.llm_manager.generate_response(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3  # Low temperature for deterministic rewrites
            )
            
            # Extract just the rewritten question
            rewritten = response.strip()
            
            # Remove common prefixes if LLM includes them
            prefixes_to_remove = [
                'rewritten question:',
                'rewritten:',
                'standalone question:',
                'here is the rewritten question:',
                'here\'s the rewritten question:'
            ]
            
            rewritten_lower = rewritten.lower()
            for prefix in prefixes_to_remove:
                if rewritten_lower.startswith(prefix):
                    rewritten = rewritten[len(prefix):].strip()
                    break
            
            return rewritten
            
        except Exception as e:
            print(f"⚠️ Error during LLM rewriting: {e}")
            return ""
    
    def get_expansion_map(self, 
                         original_query: str,
                         rewritten_query: str) -> Dict[str, str]:
        """
        Get mapping of what was expanded/replaced.
        
        Args:
            original_query: Original user question
            rewritten_query: Rewritten question
            
        Returns:
            Dictionary mapping original terms to replacements
        """
        # Simple heuristic: find pronouns in original and what they became
        expansion_map = {}
        
        original_words = original_query.lower().split()
        rewritten_words = rewritten_query.lower().split()
        
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
        
        for pronoun in pronouns:
            if pronoun in original_words and pronoun not in rewritten_words:
                # Pronoun was replaced - try to find what it became
                expansion_map[pronoun] = "expanded"
        
        return expansion_map if expansion_map else None
    
    def batch_rewrite(self, 
                     queries: List[str],
                     conversation_history: List[Dict]) -> List[str]:
        """
        Rewrite multiple queries in batch.
        
        Args:
            queries: List of queries to rewrite
            conversation_history: Conversation context
            
        Returns:
            List of rewritten queries
        """
        return [
            self.rewrite_with_context(q, conversation_history) 
            for q in queries
        ]
