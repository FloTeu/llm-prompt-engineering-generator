import logging
from typing import Optional, Type

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import OutputParserException
from pydantic import BaseModel

from llm_prompting_gen.models.prompt_engineering import PEMessages, PromptElements


class PromptEngineeringGenerator:
    """
    Combines Prompt Engineering dataclass with LLM.
    """

    def __init__(self, llm: BaseLanguageModel, prompt_elements: Optional[PromptElements] = None):
        self.llm = llm
        self.prompt_elements = prompt_elements or PromptElements()

    @classmethod
    def from_json(cls, file_path: str, llm: BaseLanguageModel):
        prompt_elements = PromptElements.parse_file(file_path)
        return cls(llm=llm, prompt_elements=prompt_elements)

    def _get_messages(self) -> PEMessages:
        """Transform the prompt elements to langchain message dataclass"""
        return PEMessages.from_pydantic(self.prompt_elements)

    def _get_llm_chain(self) -> LLMChain:
        """Combines chat messages with LLM and returns a LLM Chain"""
        chat_prompt: ChatPromptTemplate = self._get_messages().get_chat_prompt_template()
        return LLMChain(llm=self.llm, prompt=chat_prompt)

    def generate(self, *args, **kwargs) -> str:
        """Generates a llm str output based on few shot learning"""
        assert self.prompt_elements.is_any_set()
        llm_chain = self._get_llm_chain()
        return llm_chain.run(*args, **kwargs)


class ParsablePromptEngineeringGenerator(PromptEngineeringGenerator):
    """
    Enhances PromptEngineeringGenerator with pydantic output format.
    LLM output will be parsed to pydantic.
    Prompt element output format is ignored, if pydantic is provided.
    """

    def __init__(self, llm: BaseLanguageModel, pydantic_cls: Type[BaseModel],
                 prompt_elements: Optional[PromptElements] = None):
        super().__init__(llm=llm, prompt_elements=prompt_elements)
        # Set up a parser
        self.output_parser = PydanticOutputParser(pydantic_object=pydantic_cls)
        self.prompt_elements.output_format = self.output_parser.get_format_instructions()

    @classmethod
    def from_json(cls, file_path: str, llm: BaseLanguageModel, pydantic_cls: Type[BaseModel]):
        prompt_elements = PromptElements.parse_file(file_path)
        return cls(llm=llm, pydantic_cls=pydantic_cls, prompt_elements=prompt_elements)

    def generate(self, *args, **kwargs) -> BaseModel:
        """Generates a pydantic parsed object"""
        llm_output = super().generate(*args, **kwargs)
        try:
            # return parsed output
            return self.output_parser.parse(llm_output)
        except OutputParserException:
            logging.warning("Could not parse llm output to pydantic class. Retry...")
            # Retry to get right format
            retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.output_parser, llm=self.llm)
            _input = self._get_llm_chain().prompt.format_prompt(*args, **kwargs)
            return retry_parser.parse_with_prompt(llm_output, _input)
