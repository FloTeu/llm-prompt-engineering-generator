from typing import List
from pydantic import BaseModel, Field

class KeywordExtractorOutput(BaseModel):
    """LLM output format for keyword extraction"""
    main_theme: str = Field(description="Overarching theme of the text")
    keywords: List[str] = Field(description="List of keywords extracted from text")
