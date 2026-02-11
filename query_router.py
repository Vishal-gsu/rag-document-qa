"""
Query Router Module
Intelligently routes queries to appropriate document collections.
Detects query intent and selects relevant collections for search.
"""
from typing import List, Dict, Optional, Set
from llm_manager import LLMManager


class QueryRouter:
    """
    Routes queries to appropriate document collections based on intent detection.
    Uses heuristics and optional LLM-based classification.
    """
    
    # Intent patterns for heuristic routing
    INTENT_PATTERNS = {
        'research_papers': {
            'keywords': [
                'paper', 'research', 'study', 'experiment', 'methodology',
                'abstract', 'conclusion', 'findings', 'algorithm', 'model',
                'neural network', 'transformer', 'bert', 'gpt', 'machine learning',
                'deep learning', 'nlp', 'computer vision', 'ai', 'artificial intelligence',
                'citation', 'author', 'journal', 'conference', 'arxiv', 'doi'
            ],
            'phrases': [
                'according to the paper',
                'research on',
                'study about',
                'what does the paper say',
                'in the abstract',
                'experimental results',
                'proposed method',
                'state of the art',
                'recent advances in'
            ]
        },
        'resumes': {
            'keywords': [
                'resume', 'cv', 'experience', 'skills', 'education',
                'qualification', 'work history', 'employment', 'job',
                'candidate', 'applicant', 'hire', 'recruit', 'portfolio',
                'python', 'java', 'javascript', 'aws', 'docker', 'kubernetes',
                'react', 'node', 'sql', 'git', 'agile', 'scrum',
                'certification', 'degree', 'bachelor', 'master', 'phd',
                'projects', 'achievements', 'responsibilities'
            ],
            'phrases': [
                'who has experience in',
                'find candidates with',
                'worked on',
                'years of experience',
                'proficient in',
                'skilled in',
                'expertise in',
                'background in',
                'show me resumes',
                'find people who'
            ]
        },
        'textbooks': {
            'keywords': [
                'textbook', 'chapter', 'section', 'lesson', 'tutorial',
                'learning', 'course', 'lecture', 'exercises', 'practice',
                'fundamentals', 'basics', 'introduction', 'guide', 'handbook',
                'examples', 'problems', 'solutions', 'key terms', 'glossary',
                'summary', 'review questions', 'learning objectives'
            ],
            'phrases': [
                'learn about',
                'how to',
                'explain the concept',
                'introduction to',
                'basics of',
                'chapter on',
                'practice problems',
                'step by step',
                'beginner guide',
                'tutorial on'
            ]
        },
        'general_docs': {
            'keywords': [
                'document', 'file', 'content', 'information', 'data',
                'text', 'note', 'memo', 'report', 'summary'
            ],
            'phrases': [
                'find information about',
                'what does it say',
                'search for',
                'look for'
            ]
        }
    }
    
    def __init__(self, 
                 use_llm: bool = False,
                 llm_manager: LLMManager = None,
                 default_collections: List[str] = None):
        """
        Initialize query router.
        
        Args:
            use_llm: Use LLM for intent classification (more accurate but slower)
            llm_manager: LLM manager for classification
            default_collections: Default collections to search if no intent detected
        """
        self.use_llm = use_llm
        self.llm_manager = llm_manager
        self.default_collections = default_collections or ['research_papers', 'resumes', 'textbooks', 'general_docs']
    
    def route_query(self, 
                    query: str,
                    available_collections: List[str] = None,
                    max_collections: int = None) -> List[str]:
        """
        Route query to appropriate collections.
        
        Args:
            query: User's query
            available_collections: List of available collection names
            max_collections: Maximum collections to return (None = all matches)
            
        Returns:
            List of collection names to search
        """
        if not query or not query.strip():
            return self.default_collections
        
        available = available_collections or self.default_collections
        
        # Use LLM-based routing if enabled
        if self.use_llm and self.llm_manager:
            return self._llm_route(query, available, max_collections)
        
        # Otherwise use heuristic routing
        return self._heuristic_route(query, available, max_collections)
    
    def _heuristic_route(self, 
                        query: str,
                        available_collections: List[str],
                        max_collections: int = None) -> List[str]:
        """
        Route query using heuristic pattern matching.
        
        Args:
            query: User's query
            available_collections: Available collections
            max_collections: Max collections to return
            
        Returns:
            List of collection names
        """
        query_lower = query.lower()
        scores = {}
        
        # Score each collection type
        for collection_type, patterns in self.INTENT_PATTERNS.items():
            if collection_type not in available_collections:
                continue
            
            score = 0
            
            # Check keywords (1 point each)
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    score += 1
            
            # Check phrases (2 points each - stronger signal)
            for phrase in patterns['phrases']:
                if phrase in query_lower:
                    score += 2
            
            scores[collection_type] = score
        
        # Get collections with non-zero scores
        matched_collections = [
            coll for coll, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if score > 0
        ]
        
        # If no matches or very low confidence, search relevant collections
        if not matched_collections:
            matched_collections = available_collections
        elif max(scores.values()) < 3:  # Low confidence threshold (< 3 points)
            # Low confidence - include related collections to avoid missing results
            # Keep matched ones but add 'general_docs' and 'textbooks' as fallback
            fallback_collections = ['general_docs', 'textbooks'] 
            for coll in fallback_collections:
                if coll in available_collections and coll not in matched_collections:
                    matched_collections.append(coll)
        
        # Limit to max_collections if specified
        if max_collections and len(matched_collections) > max_collections:
            matched_collections = matched_collections[:max_collections]
        
        return matched_collections
    
    def _llm_route(self,
                   query: str,
                   available_collections: List[str],
                   max_collections: int = None) -> List[str]:
        """
        Route query using LLM-based classification.
        
        Args:
            query: User's query
            available_collections: Available collections
            max_collections: Max collections to return
            
        Returns:
            List of collection names
        """
        # Build classification prompt
        collections_str = ", ".join(available_collections)
        
        prompt = f"""You are a query routing assistant. Given a user's query, determine which document collections are most relevant.

Available collections:
- research_papers: Academic papers, research studies, scientific articles
- resumes: CVs, candidate profiles, work experience, skills
- textbooks: Educational content, tutorials, learning materials, chapters
- general_docs: General documents, notes, reports

User query: "{query}"

Instructions:
1. Analyze the query intent
2. Select 1-3 most relevant collections
3. Output ONLY the collection names as a comma-separated list
4. If the query is general or unclear, output all collections

Collections to search:"""
        
        try:
            response = self.llm_manager.generate_response(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1  # Low temperature for deterministic classification
            )
            
            # Parse response
            selected = [
                c.strip() 
                for c in response.lower().replace('collections to search:', '').strip().split(',')
            ]
            
            # Filter to available collections
            selected = [c for c in selected if c in available_collections]
            
            # Fallback if parsing failed
            if not selected:
                selected = available_collections
            
            # Limit to max_collections
            if max_collections and len(selected) > max_collections:
                selected = selected[:max_collections]
            
            return selected
            
        except Exception as e:
            print(f"⚠️ LLM routing failed: {e}. Falling back to heuristics.")
            return self._heuristic_route(query, available_collections, max_collections)
    
    def get_intent_confidence(self, query: str) -> Dict[str, float]:
        """
        Get confidence scores for each collection type.
        
        Args:
            query: User's query
            
        Returns:
            Dictionary mapping collection types to confidence scores (0-1)
        """
        query_lower = query.lower()
        scores = {}
        max_possible = 10  # Reasonable upper bound for scoring
        
        for collection_type, patterns in self.INTENT_PATTERNS.items():
            score = 0
            
            # Count keyword matches
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    score += 1
            
            # Count phrase matches (weighted higher)
            for phrase in patterns['phrases']:
                if phrase in query_lower:
                    score += 2
            
            # Normalize to 0-1 range
            confidence = min(score / max_possible, 1.0)
            scores[collection_type] = confidence
        
        return scores
    
    def explain_routing(self, query: str, selected_collections: List[str]) -> str:
        """
        Explain why certain collections were selected.
        
        Args:
            query: User's query
            selected_collections: Collections that were selected
            
        Returns:
            Human-readable explanation
        """
        confidences = self.get_intent_confidence(query)
        
        explanation_parts = [f"Query: '{query}'", ""]
        explanation_parts.append("Collection confidence scores:")
        
        for collection_type in self.INTENT_PATTERNS.keys():
            confidence = confidences.get(collection_type, 0.0)
            selected = "✓" if collection_type in selected_collections else "✗"
            explanation_parts.append(f"  {selected} {collection_type}: {confidence:.2f}")
        
        explanation_parts.append("")
        explanation_parts.append(f"Searching: {', '.join(selected_collections)}")
        
        return "\n".join(explanation_parts)
    
    def add_custom_pattern(self, 
                          collection_type: str,
                          keywords: List[str] = None,
                          phrases: List[str] = None):
        """
        Add custom routing patterns.
        
        Args:
            collection_type: Collection to add patterns for
            keywords: Additional keywords
            phrases: Additional phrases
        """
        if collection_type not in self.INTENT_PATTERNS:
            self.INTENT_PATTERNS[collection_type] = {
                'keywords': [],
                'phrases': []
            }
        
        if keywords:
            self.INTENT_PATTERNS[collection_type]['keywords'].extend(keywords)
        
        if phrases:
            self.INTENT_PATTERNS[collection_type]['phrases'].extend(phrases)
