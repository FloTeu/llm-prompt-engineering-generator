import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import OutputParserException
from pydantic import BaseModel

from llm_few_shot_gen.models.generator import FewShotGenerationMessages
from llm_few_shot_gen.models.prompt_engineering import PEMessages, PromptElements

class FewShotGenerator:
    """
    Few shot generator class. Combines Prompt Engineering dataclass with LLM.
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


class ParsableFewShotGenerator(FewShotGenerator):
    """
    Enhances FewShotGenerator with pydantic output format.
    LLM output will be parsed to pydantic.
    Prompt element output format is ignored, if pydantic is provided.
    """

    def __init__(self, llm: BaseLanguageModel, pydantic_cls: Type[BaseModel], prompt_elements: Optional[PromptElements] = None):
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



class BaseFewShotGenerator(ABC):
    """
    Abstract few shot generator class. Subclasses have the ability to generate text from small amount of examples.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.messages: FewShotGenerationMessages = FewShotGenerationMessages()

    @abstractmethod
    def _set_system_instruction(self):
        """Extends self.messages with instruction and type SystemMessagePromptTemplate"""
        raise NotImplementedError

    @abstractmethod
    def _set_context(self):
        """Extends self.messages with context and type List[SystemMessagePromptTemplate]"""
        raise NotImplementedError

    @abstractmethod
    def set_few_shot_examples(self, few_shot_examples: List[str]):
        """Extends self.messages with few_shot_examples and type List[SystemMessagePromptTemplate]"""
        raise NotImplementedError

    @abstractmethod
    def _set_io_prompt(self):
        """Extends self.messages with human message for output creation and type HumanMessagePromptTemplate"""
        raise NotImplementedError

    def _fill_messages(self):
        if not self.messages.few_shot_examples:
            raise ValueError("Prompt examples are not yet provided")
        if not self.messages.instruction:
            self._set_system_instruction()
        if not self.messages.context:
            self._set_context()
        if not self.messages.io_prompt:
            self._set_io_prompt()

    def _get_llm_chain(self) -> LLMChain:
        """Combines chat messages with LLM and returns a LLM Chain"""
        chat_prompt: ChatPromptTemplate = self.messages.get_chat_prompt_template()
        return LLMChain(llm=self.llm, prompt=chat_prompt)

    def generate(self, *args, **kwargs) -> str:
        """Generates a llm str output based on few shot learning"""
        self._fill_messages()
        llm_chain = self._get_llm_chain()
        return llm_chain.run(*args, **kwargs)


class BaseParsableFewShotGenerator(BaseFewShotGenerator):
    """
    Abstract few shot generator class.
    Subclasses have the ability to generate pydantic object from small amount of examples.
    """

    def __init__(self, llm: BaseLanguageModel, pydantic_cls: BaseModel):
        super().__init__(llm=llm)
        # Set up a parser
        self.output_parser = PydanticOutputParser(pydantic_object=pydantic_cls)

    def is_io_prompt_parsable(self):
        """Check if format_instructions is already provided in io_prompt"""
        return "format_instructions" in self.messages.io_prompt.prompt.partial_variables

    def make_io_prompt_parsable(self):
        """Transforms a io_prompt to a parsable prompt (if its not already parsable)"""
        if not self.is_io_prompt_parsable():
            # Extend template with format instruction
            self.messages.io_prompt.prompt.template = self.messages.io_prompt.prompt.template + "\n{format_instructions}"
            # Set partial variable
            self.messages.io_prompt.prompt.partial_variables = {"format_instructions": self.output_parser.get_format_instructions()}

    def _get_parsable_chat_prompt_template(self) -> ChatPromptTemplate:
        """Extends chat prompt template form self.messages with output format instruction"""
        self.make_io_prompt_parsable()
        # Return chat prompt template with output format instruction
        return self.messages.get_chat_prompt_template()

    def _get_llm_chain(self) -> LLMChain:
        """Combines parsable chat messages with LLM and returns a LLM Chain"""
        return LLMChain(llm=self.llm, prompt=self._get_parsable_chat_prompt_template())

    def generate(self, *args, **kwargs) -> BaseModel:
        """Generates a pydantic parsed object based on few shot learning"""
        # make sure all messages ar set
        self._fill_messages()
        # get llm chain by llm and chat messages
        llm_chain = self._get_llm_chain()
        # Model output hopfully parsable to pydantic format
        output_content: str = llm_chain.run(*args, **kwargs)
        try:
            # return parsed output
            return self.output_parser.parse(output_content)
        except OutputParserException:
            logging.warning("Could not parse llm output to pydantic class. Retry...")
            # Retry to get right format
            retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.output_parser, llm=self.llm)
            _input = llm_chain.prompt.format_prompt(*args, **kwargs)
            return retry_parser.parse_with_prompt(output_content, _input)
