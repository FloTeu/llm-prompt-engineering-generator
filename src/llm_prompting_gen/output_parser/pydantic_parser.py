from typing import Type
from pydantic import BaseModel
from langchain.output_parsers.pydantic import PydanticOutputParser as LangChainPydanticOutputParser


class PydanticOutputParser(LangChainPydanticOutputParser):
    """Parse an output using a pydantic model.
        Uses pydantic v2 instead of v1 like langchain
    """

    pydantic_object: Type[BaseModel]