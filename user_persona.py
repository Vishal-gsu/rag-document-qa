"""
User Persona Module
Manages user expertise profiles and adjusts RAG behavior accordingly.
"""
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PersonaProfile:
    """User persona profile with RAG configuration."""
    name: str
    description: str
    top_k: int
    temperature: float
    max_tokens: int
    system_prompt: str
    retrieval_strategy: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
            'retrieval_strategy': self.retrieval_strategy
        }


class UserPersona:
    """
    Manages user personas for adaptive RAG behavior.
    Different personas get different retrieval and generation strategies.
    """
    
    # Predefined persona profiles
    PERSONAS = {
        'beginner': PersonaProfile(
            name='Beginner',
            description='New to the topic, needs detailed explanations with examples',
            top_k=3,  # Fewer sources to avoid overwhelming
            temperature=0.7,  # More creative explanations
            max_tokens=600,  # Longer, more detailed answers
            system_prompt="""You are a patient and helpful teacher explaining concepts to beginners.

Guidelines:
- Use simple, clear language without jargon
- Provide concrete examples and analogies
- Break down complex topics into digestible parts
- Define technical terms when you use them
- Encourage learning with a supportive tone
- Use bullet points and numbered lists for clarity

Format your answers to be educational and easy to follow.""",
            retrieval_strategy='broad'  # Cast wider net for context
        ),
        
        'intermediate': PersonaProfile(
            name='Intermediate',
            description='Has basic knowledge, wants practical insights and connections',
            top_k=5,  # Standard retrieval
            temperature=0.5,  # Balanced creativity
            max_tokens=500,  # Standard answer length
            system_prompt="""You are a knowledgeable guide helping someone with intermediate understanding.

Guidelines:
- Assume basic familiarity with core concepts
- Focus on practical applications and use cases
- Make connections between related topics
- Provide comparisons and trade-offs
- Use technical terms appropriately with brief explanations
- Balance depth with clarity

Format your answers to be informative and actionable.""",
            retrieval_strategy='balanced'  # Standard retrieval
        ),
        
        'expert': PersonaProfile(
            name='Expert',
            description='Advanced user, wants technical depth and precise details',
            top_k=7,  # More sources for comprehensive coverage
            temperature=0.3,  # More focused, technical responses
            max_tokens=400,  # Concise, information-dense answers
            system_prompt="""You are a technical expert providing precise, detailed information to an advanced user.

Guidelines:
- Use technical terminology without over-explaining
- Focus on implementation details and edge cases
- Cite specific methodologies, algorithms, or metrics
- Highlight nuances and advanced considerations
- Be concise but comprehensive
- Reference research papers or technical sources when relevant

Format your answers to be technically accurate and information-dense.""",
            retrieval_strategy='precise'  # More focused retrieval
        ),
        
        'researcher': PersonaProfile(
            name='Researcher',
            description='Academic/research focus, wants citations and methodology',
            top_k=10,  # Maximum sources for comprehensive literature review
            temperature=0.2,  # Very focused, factual responses
            max_tokens=700,  # Longer for detailed analysis
            system_prompt="""You are an academic research assistant providing scholarly analysis.

Guidelines:
- Emphasize methodology, experimental design, and results
- Provide citations and source attribution
- Discuss limitations and future research directions
- Compare different approaches and their trade-offs
- Use formal academic language
- Highlight key findings and contributions
- Note consensus vs. open questions in the field

Format your answers as scholarly summaries with clear source attribution.""",
            retrieval_strategy='comprehensive'  # Maximum retrieval depth
        )
    }
    
    def __init__(self, default_persona: str = 'intermediate'):
        """
        Initialize user persona manager.
        
        Args:
            default_persona: Default persona to use
        """
        self.current_persona = default_persona
        self.custom_personas = {}
    
    def get_profile(self, persona_name: str = None) -> PersonaProfile:
        """
        Get persona profile by name.
        
        Args:
            persona_name: Name of persona (beginner/intermediate/expert/researcher)
            
        Returns:
            PersonaProfile object
        """
        persona_name = persona_name or self.current_persona
        
        # Check custom personas first
        if persona_name in self.custom_personas:
            return self.custom_personas[persona_name]
        
        # Fall back to predefined personas
        if persona_name in self.PERSONAS:
            return self.PERSONAS[persona_name]
        
        # Default to intermediate
        return self.PERSONAS['intermediate']
    
    def set_persona(self, persona_name: str) -> bool:
        """
        Set current active persona.
        
        Args:
            persona_name: Name of persona to activate
            
        Returns:
            True if successful
        """
        if persona_name in self.PERSONAS or persona_name in self.custom_personas:
            self.current_persona = persona_name
            return True
        return False
    
    def list_personas(self) -> Dict[str, str]:
        """
        List all available personas with descriptions.
        
        Returns:
            Dictionary mapping persona names to descriptions
        """
        personas = {}
        
        # Add predefined personas
        for name, profile in self.PERSONAS.items():
            personas[name] = profile.description
        
        # Add custom personas
        for name, profile in self.custom_personas.items():
            personas[name] = profile.description
        
        return personas
    
    def add_custom_persona(self, 
                          name: str,
                          description: str,
                          top_k: int = 5,
                          temperature: float = 0.5,
                          max_tokens: int = 500,
                          system_prompt: str = None,
                          retrieval_strategy: str = 'balanced') -> None:
        """
        Add a custom user persona.
        
        Args:
            name: Unique persona name
            description: Description of this persona
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            max_tokens: Maximum response tokens
            system_prompt: Custom system prompt
            retrieval_strategy: Retrieval approach (broad/balanced/precise/comprehensive)
        """
        if system_prompt is None:
            system_prompt = self.PERSONAS['intermediate'].system_prompt
        
        profile = PersonaProfile(
            name=name,
            description=description,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            retrieval_strategy=retrieval_strategy
        )
        
        self.custom_personas[name] = profile
    
    def get_retrieval_params(self, persona_name: str = None) -> Dict:
        """
        Get retrieval parameters for a persona.
        
        Args:
            persona_name: Persona to get params for
            
        Returns:
            Dictionary with top_k and other retrieval settings
        """
        profile = self.get_profile(persona_name)
        
        return {
            'top_k': profile.top_k,
            'retrieval_strategy': profile.retrieval_strategy
        }
    
    def get_generation_params(self, persona_name: str = None) -> Dict:
        """
        Get generation parameters for a persona.
        
        Args:
            persona_name: Persona to get params for
            
        Returns:
            Dictionary with temperature, max_tokens, system_prompt
        """
        profile = self.get_profile(persona_name)
        
        return {
            'temperature': profile.temperature,
            'max_tokens': profile.max_tokens,
            'system_prompt': profile.system_prompt
        }
    
    def get_current_profile(self) -> PersonaProfile:
        """Get currently active persona profile."""
        return self.get_profile(self.current_persona)
    
    def adapt_to_query_complexity(self, query: str, base_persona: str = None) -> PersonaProfile:
        """
        Dynamically adapt persona based on query complexity.
        
        Args:
            query: User's query
            base_persona: Starting persona (optional)
            
        Returns:
            Adapted PersonaProfile
        """
        base_profile = self.get_profile(base_persona)
        
        # Simple heuristics for query complexity
        query_lower = query.lower()
        
        # Technical indicators
        technical_terms = ['algorithm', 'implementation', 'architecture', 'optimize', 
                          'performance', 'complexity', 'methodology', 'framework']
        beginner_terms = ['what is', 'explain', 'how does', 'why', 'basics', 'introduction']
        
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        beginner_count = sum(1 for term in beginner_terms if term in query_lower)
        
        # Adjust top_k based on query length (longer query = more specific, needs more context)
        word_count = len(query.split())
        
        if word_count > 20 or technical_count >= 2:
            # Complex query: increase top_k
            adjusted_top_k = min(base_profile.top_k + 2, 10)
        elif beginner_count >= 1:
            # Beginner query: decrease top_k for simplicity
            adjusted_top_k = max(base_profile.top_k - 1, 3)
        else:
            adjusted_top_k = base_profile.top_k
        
        # Create adapted profile
        adapted = PersonaProfile(
            name=f"{base_profile.name} (Adapted)",
            description=base_profile.description,
            top_k=adjusted_top_k,
            temperature=base_profile.temperature,
            max_tokens=base_profile.max_tokens,
            system_prompt=base_profile.system_prompt,
            retrieval_strategy=base_profile.retrieval_strategy
        )
        
        return adapted
