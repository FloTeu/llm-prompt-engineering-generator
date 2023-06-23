from typing import List
from pydantic import BaseModel, Field


# Define your desired data structure.
class ImagePromptOutputModel(BaseModel):
    few_shot_styles_artists: List[str] = Field(description="Styles or artists of the example prompts")
    image_prompts: List[str] = Field(description="List of text-to-image prompts")
