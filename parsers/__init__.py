"""
Document Parsers Package
Specialized parsers for different document types.
"""
from .base_parser import BaseParser
from .generic_parser import GenericParser
from .research_paper_parser import ResearchPaperParser
from .resume_parser import ResumeParser
from .textbook_parser import TextbookParser

__all__ = [
    'BaseParser',
    'GenericParser',
    'ResearchPaperParser',
    'ResumeParser',
    'TextbookParser'
]
