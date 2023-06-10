from abc import abstractmethod, ABC
from typing import List

from langchain.base_language import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate

from llm_few_shot_gen.data_classes import FewShotGenerationMessages


class AbstractFewShotGenerator(ABC):
    """
    Abstract few shot generator class. Subclasses have the ability to generate text from small amount of examples.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.messages: FewShotGenerationMessages = FewShotGenerationMessages()

    @abstractmethod
    def _set_system_instruction(self):
        """Extends self.messages with instruction_message"""
        raise NotImplementedError

    @abstractmethod
    def set_context(self):
        """Extends self.messages with context"""
        raise NotImplementedError

    @abstractmethod
    def set_few_shot_examples(self, few_shot_examples: List[str]):
        """Extends self.messages with few_shot_examples"""
        raise NotImplementedError

    @abstractmethod
    def _set_human_message(self):
        """Extends self.messages with human message for output creation"""
        raise NotImplementedError

    def _get_llm_chain(self) -> LLMChain:
        if not self.messages.are_few_shot_examples_set():
            raise ValueError("Prompt examples are not yet provided")
        if not self.messages.is_instruction_known():
            self._set_system_instruction()
        if not self.messages.is_context_known():
            self.set_context()
        if not self.messages.is_human_message_set():
            self._set_human_message()

        chat_prompt: ChatPromptTemplate = self.messages.get_chat_prompt_template()
        return LLMChain(llm=self.llm, prompt=chat_prompt)

    def generate(self, *args, **kwargs) -> str:
        llm_chain = self._get_llm_chain()
        return llm_chain.run(*args, **kwargs)

