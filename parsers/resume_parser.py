"""
Resume Parser
Specialized parser for resumes/CVs with section and skill extraction.
"""
import re
from typing import List, Dict, Set, Tuple
from .base_parser import BaseParser


class ResumeParser(BaseParser):
    """
    Parser for resumes and CVs.
    Detects sections (education, experience, skills) and extracts key information.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize resume parser."""
        super().__init__(chunk_size, chunk_overlap)
        
        # Section patterns
        self.section_patterns = [
            ('contact', [
                r'^contact\s+information\s*$',
                r'^contact\s*$',
            ]),
            ('summary', [
                r'^professional\s+summary\s*$',
                r'^summary\s*$',
                r'^objective\s*$',
                r'^profile\s*$',
            ]),
            ('education', [
                r'^education\s*$',
                r'^academic\s+background\s*$',
                r'^qualifications\s*$',
            ]),
            ('experience', [
                r'^(?:work\s+)?experience\s*$',
                r'^professional\s+experience\s*$',
                r'^employment\s+history\s*$',
                r'^work\s+history\s*$',
            ]),
            ('skills', [
                r'^skills?\s*$',
                r'^technical\s+skills\s*$',
                r'^core\s+competencies\s*$',
                r'^technologies\s*$',
            ]),
            ('projects', [
                r'^projects?\s*$',
                r'^key\s+projects\s*$',
            ]),
            ('certifications', [
                r'^certifications?\s*$',
                r'^licenses?\s*$',
            ]),
            ('achievements', [
                r'^achievements?\s*$',
                r'^accomplishments?\s*$',
                r'^awards?\s*$',
            ]),
        ]
        
        # Common technical skills to detect
        self.skill_keywords = {
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab',
            
            # Frameworks/Libraries
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            
            # Cloud/DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            
            # AI/ML
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'rag',
            'transformers', 'llm', 'neural networks',
            
            # Other
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum'
        }
    
    def parse(self, content: str, metadata: Dict) -> List[Dict]:
        """
        Parse resume with section detection and skill extraction.
        
        Args:
            content: Raw document text
            metadata: Document-level metadata
            
        Returns:
            List of chunks with section labels and extracted skills
        """
        if not content or not content.strip():
            return []
        
        # Extract contact information
        contact_info = self._extract_contact_info(content)
        
        # Extract skills
        skills = self._extract_skills(content)
        
        # Detect sections
        sections = self._detect_sections(content)
        
        # If no sections detected, fall back to generic chunking
        if not sections:
            return self._parse_as_generic(content, metadata, skills, contact_info)
        
        # Chunk each section separately
        result = []
        global_chunk_id = 0
        
        for section_name, section_text in sections:
            # Extract section-specific skills
            section_skills = self._extract_skills(section_text) if section_name == 'skills' else set()
            
            section_chunks = self._split_text(section_text)
            
            for chunk_text in section_chunks:
                chunk_metadata = self._create_chunk_metadata(
                    base_metadata=metadata,
                    chunk_id=global_chunk_id,
                    total_chunks=0,  # Will update after
                    section=section_name,
                    skills=list(skills),
                    **contact_info
                )
                
                # Add section-specific skills if in skills section
                if section_skills:
                    chunk_metadata['section_skills'] = list(section_skills)
                
                result.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                global_chunk_id += 1
        
        # Update total_chunks for all chunks
        total_chunks = len(result)
        for chunk in result:
            chunk['metadata']['total_chunks'] = total_chunks
        
        return result
    
    def _detect_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        Detect sections in resume.
        
        Args:
            content: Resume text
            
        Returns:
            List of (section_name, section_text) tuples
        """
        lines = content.split('\n')
        sections = []
        current_section = 'summary'  # Default
        current_text = []
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Check if line is a section header
            section_found = False
            for section_name, patterns in self.section_patterns:
                for pattern in patterns:
                    if re.match(pattern, line_lower, re.IGNORECASE):
                        # Save previous section
                        if current_text:
                            sections.append((current_section, '\n'.join(current_text)))
                        
                        # Start new section
                        current_section = section_name
                        current_text = []
                        section_found = True
                        break
                if section_found:
                    break
            
            if not section_found:
                current_text.append(line)
        
        # Add final section
        if current_text:
            sections.append((current_section, '\n'.join(current_text)))
        
        return sections
    
    def _extract_contact_info(self, content: str) -> Dict:
        """
        Extract contact information from resume.
        
        Args:
            content: Resume text
            
        Returns:
            Dictionary with name, email, phone (if detected)
        """
        contact = {}
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, content)
        if email_match:
            contact['email'] = email_match.group(0)
        
        # Extract phone
        phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
        phone_match = re.search(phone_pattern, content)
        if phone_match:
            contact['phone'] = phone_match.group(0)
        
        # Extract name (heuristic: first non-empty line, capitalized)
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 4 and line[0].isupper():
                # Likely a name if short and capitalized
                contact['name'] = line
                break
        
        return contact
    
    def _extract_skills(self, content: str) -> Set[str]:
        """
        Extract technical skills from resume.
        
        Args:
            content: Resume text
            
        Returns:
            Set of detected skills
        """
        skills = set()
        content_lower = content.lower()
        
        # Look for skills in content
        for skill in self.skill_keywords:
            # Use word boundaries for single-word skills
            if ' ' not in skill:
                pattern = r'\b' + re.escape(skill) + r'\b'
            else:
                pattern = re.escape(skill)
            
            if re.search(pattern, content_lower):
                skills.add(skill)
        
        return skills
    
    def _parse_as_generic(self, content: str, metadata: Dict, 
                         skills: Set[str], contact_info: Dict) -> List[Dict]:
        """Fallback to generic parsing if no sections detected."""
        chunks = self._split_text(content)
        
        result = []
        total_chunks = len(chunks)
        
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = self._create_chunk_metadata(
                base_metadata=metadata,
                chunk_id=i,
                total_chunks=total_chunks,
                section='unknown',
                skills=list(skills),
                **contact_info
            )
            
            result.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return result
