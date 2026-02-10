"""
Textbook Parser
Specialized parser for textbooks with chapter and section detection.
"""
import re
from typing import List, Dict, Tuple, Optional
from .base_parser import BaseParser


class TextbookParser(BaseParser):
    """
    Parser for educational textbooks.
    Detects chapters, sections, learning objectives, and exercises.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize textbook parser."""
        super().__init__(chunk_size, chunk_overlap)
        
        # Chapter patterns
        self.chapter_patterns = [
            r'^chapter\s+(\d+)[:\s]',
            r'^ch\.?\s+(\d+)[:\s]',
            r'^(\d+)\.\s+[A-Z]',  # "1. Introduction"
        ]
        
        # Section patterns
        self.section_patterns = [
            r'^(\d+\.\d+)\s+',  # "1.1 Introduction"
            r'^section\s+(\d+\.\d+)',
        ]
        
        # Special section types
        self.special_sections = {
            'learning_objectives': [
                r'^learning\s+objectives?\s*[:.]?',
                r'^objectives?\s*[:.]?',
                r'^in\s+this\s+chapter',
            ],
            'summary': [
                r'^summary\s*[:.]?',
                r'^chapter\s+summary',
                r'^key\s+points',
            ],
            'exercises': [
                r'^exercises?\s*[:.]?',
                r'^problems?\s*[:.]?',
                r'^practice\s+problems',
                r'^review\s+questions',
            ],
            'key_terms': [
                r'^key\s+terms\s*[:.]?',
                r'^glossary\s*[:.]?',
                r'^vocabulary\s*[:.]?',
            ],
        }
    
    def parse(self, content: str, metadata: Dict) -> List[Dict]:
        """
        Parse textbook with chapter and section detection.
        
        Args:
            content: Raw document text
            metadata: Document-level metadata
            
        Returns:
            List of chunks with chapter/section labels
        """
        if not content or not content.strip():
            return []
        
        # Detect chapter structure
        chapters = self._detect_chapters(content)
        
        # If no chapters detected, try section-based parsing
        if not chapters:
            sections = self._detect_sections_only(content)
            if sections:
                return self._parse_sections(sections, metadata)
            else:
                return self._parse_as_generic(content, metadata)
        
        # Parse chapters
        result = []
        global_chunk_id = 0
        
        for chapter_num, chapter_text in chapters:
            # Detect sections within chapter
            sections = self._detect_sections_within_chapter(chapter_text)
            
            if sections:
                # Parse with sections
                for section_id, section_type, section_text in sections:
                    section_chunks = self._split_text(section_text)
                    
                    for chunk_text in section_chunks:
                        chunk_metadata = self._create_chunk_metadata(
                            base_metadata=metadata,
                            chunk_id=global_chunk_id,
                            total_chunks=0,  # Will update after
                            chapter=chapter_num,
                            section=section_id,
                            section_type=section_type
                        )
                        
                        result.append({
                            'text': chunk_text,
                            'metadata': chunk_metadata
                        })
                        
                        global_chunk_id += 1
            else:
                # Chapter without sections
                chapter_chunks = self._split_text(chapter_text)
                
                for chunk_text in chapter_chunks:
                    chunk_metadata = self._create_chunk_metadata(
                        base_metadata=metadata,
                        chunk_id=global_chunk_id,
                        total_chunks=0,
                        chapter=chapter_num,
                        section='main'
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
    
    def _detect_chapters(self, content: str) -> List[Tuple[str, str]]:
        """
        Detect chapters in textbook.
        
        Args:
            content: Textbook text
            
        Returns:
            List of (chapter_number, chapter_text) tuples
        """
        lines = content.split('\n')
        chapters = []
        current_chapter = None
        current_text = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if line is a chapter marker
            chapter_found = False
            for pattern in self.chapter_patterns:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    # Save previous chapter
                    if current_chapter is not None and current_text:
                        chapters.append((current_chapter, '\n'.join(current_text)))
                    
                    # Start new chapter
                    current_chapter = match.group(1) if match.lastindex else str(len(chapters) + 1)
                    current_text = [line]  # Include chapter title
                    chapter_found = True
                    break
            
            if not chapter_found and current_chapter is not None:
                current_text.append(line)
        
        # Add final chapter
        if current_chapter is not None and current_text:
            chapters.append((current_chapter, '\n'.join(current_text)))
        
        return chapters
    
    def _detect_sections_within_chapter(self, chapter_text: str) -> List[Tuple[str, str, str]]:
        """
        Detect sections within a chapter.
        
        Args:
            chapter_text: Text of a chapter
            
        Returns:
            List of (section_id, section_type, section_text) tuples
        """
        lines = chapter_text.split('\n')
        sections = []
        current_section = '0'
        current_type = 'main'
        current_text = []
        
        for line in lines:
            line_stripped = line.strip().lower()
            
            # Check for special sections
            section_found = False
            for special_type, patterns in self.special_sections.items():
                for pattern in patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        # Save previous section
                        if current_text:
                            sections.append((current_section, current_type, '\n'.join(current_text)))
                        
                        # Start new special section
                        current_section = special_type
                        current_type = special_type
                        current_text = []
                        section_found = True
                        break
                if section_found:
                    break
            
            # Check for numbered sections
            if not section_found:
                for pattern in self.section_patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        # Save previous section
                        if current_text:
                            sections.append((current_section, current_type, '\n'.join(current_text)))
                        
                        # Start new numbered section
                        current_section = match.group(1)
                        current_type = 'content'
                        current_text = [line]
                        section_found = True
                        break
            
            if not section_found:
                current_text.append(line)
        
        # Add final section
        if current_text:
            sections.append((current_section, current_type, '\n'.join(current_text)))
        
        return sections
    
    def _detect_sections_only(self, content: str) -> List[Tuple[str, str]]:
        """Detect sections when no chapters are found."""
        # Similar to chapter detection but for sections
        lines = content.split('\n')
        sections = []
        current_section = '1'
        current_text = []
        
        for line in lines:
            section_found = False
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    if current_text:
                        sections.append((current_section, '\n'.join(current_text)))
                    current_section = match.group(1)
                    current_text = [line]
                    section_found = True
                    break
            
            if not section_found:
                current_text.append(line)
        
        if current_text:
            sections.append((current_section, '\n'.join(current_text)))
        
        return sections
    
    def _parse_sections(self, sections: List[Tuple[str, str]], metadata: Dict) -> List[Dict]:
        """Parse when only sections (no chapters) are detected."""
        result = []
        global_chunk_id = 0
        
        for section_id, section_text in sections:
            section_chunks = self._split_text(section_text)
            
            for chunk_text in section_chunks:
                chunk_metadata = self._create_chunk_metadata(
                    base_metadata=metadata,
                    chunk_id=global_chunk_id,
                    total_chunks=0,
                    section=section_id,
                    section_type='content'
                )
                
                result.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                global_chunk_id += 1
        
        total_chunks = len(result)
        for chunk in result:
            chunk['metadata']['total_chunks'] = total_chunks
        
        return result
    
    def _parse_as_generic(self, content: str, metadata: Dict) -> List[Dict]:
        """Fallback to generic parsing if no structure detected."""
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
