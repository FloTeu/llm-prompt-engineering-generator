from typing import List
from pydantic import BaseModel, Field


# Define your desired data structure.
class ImagePromptOutputModel(BaseModel):
    few_shot_art_styles: List[str] = Field(description="Art styles of the example prompts")
    image_prompts: List[str] = Field(description="List of text-to-image prompts")
