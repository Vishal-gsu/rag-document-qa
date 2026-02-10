"""
Research Paper Parser
Specialized parser for academic papers with section detection.
"""
import re
from typing import List, Dict, Tuple
from .base_parser import BaseParser


class ResearchPaperParser(BaseParser):
    """
    Parser for academic research papers.
    Detects and labels sections: abstract, introduction, methodology, results, conclusion, references.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize research paper parser."""
        super().__init__(chunk_size, chunk_overlap)
        
        # Section patterns (ordered by typical appearance)
        self.section_patterns = [
            ('abstract', [
                r'^abstract\s*$',
                r'^\d+\.?\s*abstract\s*$',
                r'^abstract[:.]',
            ]),
            ('introduction', [
                r'^introduction\s*$',
                r'^\d+\.?\s*introduction\s*$',
                r'^1\.?\s+introduction',
            ]),
            ('related_work', [
                r'^related\s+work\s*$',
                r'^\d+\.?\s*related\s+work\s*$',
                r'^background\s*$',
                r'^\d+\.?\s*background\s*$',
            ]),
            ('methodology', [
                r'^methodology\s*$',
                r'^methods?\s*$',
                r'^\d+\.?\s*methodology\s*$',
                r'^\d+\.?\s*methods?\s*$',
                r'^approach\s*$',
                r'^experimental\s+setup\s*$',
            ]),
            ('results', [
                r'^results?\s*$',
                r'^\d+\.?\s*results?\s*$',
                r'^experiments?\s*$',
                r'^\d+\.?\s*experiments?\s*$',
                r'^evaluation\s*$',
            ]),
            ('discussion', [
                r'^discussion\s*$',
                r'^\d+\.?\s*discussion\s*$',
                r'^analysis\s*$',
            ]),
            ('conclusion', [
                r'^conclusions?\s*$',
                r'^\d+\.?\s*conclusions?\s*$',
                r'^summary\s*$',
                r'^future\s+work\s*$',
            ]),
            ('references', [
                r'^references\s*$',
                r'^bibliography\s*$',
                r'^\d+\.?\s*references\s*$',
            ]),
        ]
    
    def parse(self, content: str, metadata: Dict) -> List[Dict]:
        """
        Parse research paper with section detection.
        
        Args:
            content: Raw document text
            metadata: Document-level metadata
            
        Returns:
            List of chunks with section labels
        """
        if not content or not content.strip():
            return []
        
        # Detect sections
        sections = self._detect_sections(content)
        
        # If no sections detected, fall back to generic chunking
        if not sections:
            return self._parse_as_generic(content, metadata)
        
        # Extract metadata from paper
        paper_metadata = self._extract_paper_metadata(content)
        
        # Chunk each section separately
        result = []
        global_chunk_id = 0
        
        for section_name, section_text in sections:
            section_chunks = self._split_text(section_text)
            
            for chunk_text in section_chunks:
                chunk_metadata = self._create_chunk_metadata(
                    base_metadata=metadata,
                    chunk_id=global_chunk_id,
                    total_chunks=0,  # Will update after
                    section=section_name,
                    **paper_metadata
                )
                
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
        Detect sections in research paper.
        
        Args:
            content: Paper text
            
        Returns:
            List of (section_name, section_text) tuples
        """
        lines = content.split('\n')
        sections = []
        current_section = 'introduction'  # Default
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
    
    def _extract_paper_metadata(self, content: str) -> Dict:
        """
        Extract metadata specific to research papers.
        
        Args:
            content: Paper text
            
        Returns:
            Dictionary with title, authors, year (if detected)
        """
        metadata = {}
        
        # Extract title (usually first non-empty line or largest header)
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 10:
                metadata['title'] = line[:200]  # Limit length
                break
        
        # Extract year (look for 4-digit year in first 1000 chars)
        year_match = re.search(r'\b(19|20)\d{2}\b', content[:1000])
        if year_match:
            metadata['year'] = year_match.group(0)
        
        # Detect if has citations (presence of et al., [1], (Author, Year))
        has_citations = bool(
            re.search(r'\bet al\.', content[:5000], re.IGNORECASE) or
            re.search(r'\[\d+\]', content[:5000]) or
            re.search(r'\(\w+,?\s+\d{4}\)', content[:5000])
        )
        metadata['has_citations'] = has_citations
        
        return metadata
    
    def _parse_as_generic(self, content: str, metadata: Dict) -> List[Dict]:
        """Fallback to generic parsing if no sections detected."""
        chunks = self._split_text(content)
        
        result = []
        total_chunks = len(chunks)
        
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = self._create_chunk_metadata(
                base_metadata=metadata,
                chunk_id=i,
                total_chunks=total_chunks,
                section='unknown'
            )
            
            result.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return result
