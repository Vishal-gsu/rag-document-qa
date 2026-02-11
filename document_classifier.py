"""
Document Classifier Module
Classifies documents into types for specialized processing.
"""
import re
from pathlib import Path
from typing import Dict, List


class DocumentClassifier:
    """
    Classifies documents into types for specialized parsing.
    
    Supported types:
    - research_paper: Academic papers with abstract, references, citations
    - resume: CVs with education, experience, skills sections
    - textbook: Educational material with chapters, exercises
    - generic: Default fallback for unclassified documents
    """
    
    def __init__(self):
        """Initialize classifier with pattern dictionaries."""
        # Research paper indicators
        self.research_patterns = {
            'strong': [
                r'\babstract\b',
                r'\breferences\b',
                r'\bcitation\b',
                r'\bdoi\b',
                r'\barxiv\b',
                r'\bintroduction\b.*\bmethodology\b.*\bresults\b',
                r'\bconclusion\b.*\breferences\b',
                r'\bet al\.',
                r'\bpublished in\b',
            ],
            'weak': [
                r'\bfigure \d+\b',
                r'\btable \d+\b',
                r'\bsection \d+\b',
                r'\b\d{4}\b.*\b(ieee|acm|springer|elsevier)\b',
            ]
        }
        
        # Resume/CV indicators
        self.resume_patterns = {
            'strong': [
                r'\beducation\b',
                r'\bexperience\b',
                r'\bskills\b',
                r'\bcertification',
                r'\bresume\b',
                r'\bcurriculum vitae\b',
                r'\bwork history\b',
                r'\bprofessional summary\b',
            ],
            'weak': [
                r'\b(?:bachelor|master|phd|b\.?s\.?|m\.?s\.?|ph\.?d\.?)\b',
                r'\b(?:university|college|institute)\b',
                r'\b\d{4}\s*-\s*(?:\d{4}|present)\b',
                r'\bemail\b.*\bphone\b',
            ]
        }
        
        # Textbook indicators
        self.textbook_patterns = {
            'strong': [
                r'\bchapter \d+\b',
                r'\bexercise[s]?\b',
                r'\blearning objectives\b',
                r'\bisbn\b',
                r'\btable of contents\b',
                r'\bpreface\b',
                r'\bintroduction to\b',  # Added: common in educational docs
                r'\bfundamentals?\b',     # Added: "Machine Learning Fundamentals"
            ],
            'weak': [
                r'\bsummary\b',
                r'\bkey terms\b',
                r'\breview questions\b',
                r'\bglossary\b',
                r'\btypes of\b',          # Added: "Types of Machine Learning"
                r'\bcommon algorithms?\b', # Added: algorithm lists
                r'\bapplications?:\b',    # Added: application sections
                r'\bkey concepts?:\b',    # Added: concept sections
            ]
        }
    
    def classify(self, file_path: Path = None, content: str = None, 
                 metadata: Dict = None) -> str:
        """
        Classify document into a type.
        
        Args:
            file_path: Path to document file (optional)
            content: Document text content
            metadata: Additional metadata (optional)
            
        Returns:
            Document type: 'research_paper', 'resume', 'textbook', or 'generic'
        """
        if not content or len(content.strip()) < 100:
            return 'generic'
        
        # Normalize content for matching (lowercase, first 5000 chars)
        sample = content[:5000].lower()
        
        # Score each document type
        scores = {
            'research_paper': self._score_patterns(sample, self.research_patterns),
            'resume': self._score_patterns(sample, self.resume_patterns),
            'textbook': self._score_patterns(sample, self.textbook_patterns),
        }
        
        # Determine winner
        max_score = max(scores.values())
        
        # Threshold: need at least 2 points to classify (prevents false positives)
        if max_score < 2.0:
            return 'generic'
        
        # Return highest scoring type
        for doc_type, score in scores.items():
            if score == max_score:
                return doc_type
        
        return 'generic'
    
    def _score_patterns(self, text: str, patterns: Dict[str, List[str]]) -> float:
        """
        Score text against pattern dictionary.
        
        Args:
            text: Text to analyze (already lowercase)
            patterns: Dict with 'strong' and 'weak' pattern lists
            
        Returns:
            Score (strong match = 1.0, weak match = 0.5)
        """
        score = 0.0
        
        # Strong patterns (1 point each)
        for pattern in patterns.get('strong', []):
            if re.search(pattern, text, re.IGNORECASE):
                score += 1.0
        
        # Weak patterns (0.5 points each)
        for pattern in patterns.get('weak', []):
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.5
        
        return score
    
    def get_confidence(self, file_path: Path = None, content: str = None) -> Dict[str, float]:
        """
        Get classification confidence scores for all types.
        
        Args:
            file_path: Path to document file (optional)
            content: Document text content
            
        Returns:
            Dict mapping document types to confidence scores
        """
        if not content or len(content.strip()) < 100:
            return {'generic': 1.0}
        
        sample = content[:5000].lower()
        
        raw_scores = {
            'research_paper': self._score_patterns(sample, self.research_patterns),
            'resume': self._score_patterns(sample, self.resume_patterns),
            'textbook': self._score_patterns(sample, self.textbook_patterns),
            'generic': 0.0
        }
        
        # Normalize to percentages
        total = sum(raw_scores.values())
        if total == 0:
            return {'generic': 1.0}
        
        normalized = {k: v/total for k, v in raw_scores.items()}
        return normalized
    
    def explain_classification(self, content: str) -> Dict:
        """
        Explain why a document was classified a certain way.
        
        Args:
            content: Document text content
            
        Returns:
            Dict with classification and matched patterns
        """
        sample = content[:5000].lower()
        doc_type = self.classify(content=content)
        
        matched_patterns = {
            'research_paper': [],
            'resume': [],
            'textbook': []
        }
        
        # Find which patterns matched
        for category in ['research_paper', 'resume', 'textbook']:
            patterns_dict = getattr(self, f"{category.replace('_', '')}_patterns", {})
            
            for strength in ['strong', 'weak']:
                for pattern in patterns_dict.get(strength, []):
                    if re.search(pattern, sample, re.IGNORECASE):
                        matched_patterns[category].append({
                            'pattern': pattern,
                            'strength': strength
                        })
        
        return {
            'classification': doc_type,
            'confidence': self.get_confidence(content=content),
            'matched_patterns': matched_patterns
        }
