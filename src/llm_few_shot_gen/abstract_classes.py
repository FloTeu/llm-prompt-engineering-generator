from abc import abstractmethod, ABC
from typing import List

from langchain.base_language import BaseLanguageModel
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.chains import LLMChain

from llm_few_shot_gen.data_classes import FewShotGenerationMessages
from llm_few_shot_gen.constants import INSTRUCTOR_USER_NAME

class AbstractFewShotGenerator(ABC):
    """
    Abstract few shot generator class. Subclasses have the ability to generate text from small amount of examples.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.messages: FewShotGenerationMessages = FewShotGenerationMessages(instruction_message=self._get_system_instruction())

    @abstractmethod
    def _get_system_instruction(self):
        """Ininitalizes self.messages with instruction_message"""
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
        if not self.messages.is_context_known():
            self.set_context()
        if not self.messages.is_human_message_set():
            self._set_human_message()

        chat_prompt: ChatPromptTemplate = self.messages.get_chat_prompt_template()
        return LLMChain(llm=self.llm, prompt=chat_prompt)

    def generate(self, *args, **kwargs) -> str:
        llm_chain = self._get_llm_chain()
        return llm_chain.run(*args, **kwargs)

class AbstractTextToImagePromptGenerator(AbstractFewShotGenerator):
    """
    Abstract prompt generator class. Subclasses have the ability to generate text-to-image prompts.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.messages: FewShotGenerationMessages = FewShotGenerationMessages(instruction_message=self._get_system_instruction())

    def _get_system_instruction(self):
        """System message to instruct the llm model how he should act"""
        return SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant in helping me create text-to-image prompts.
            """)

    def set_few_shot_examples(self, few_shot_examples: List[str]):
        """
        Extends self.messages with prompt examples of Text-to-Image AI
        Few shot learning is implemented in this function
        OpenAI suggests to provide few shot examples with system messages and the "name" argument
        """
        messages = [SystemMessagePromptTemplate.from_template(
            "Here are some example prompts. Try to understand the underlying format of prompts in order to create new creative prompts yourself later. ",
            additional_kwargs={"name": INSTRUCTOR_USER_NAME})]
        for i, example_prompt in enumerate(few_shot_examples):
            messages.append(
                SystemMessagePromptTemplate.from_template(f'Prompt {i}: "{example_prompt}". ',
                                                          additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        self.messages.few_shot_examples = messages

    def _set_human_message(self):
        """Human message which contains the input for the prompt generation"""
        human_template = """
                            I want you to act as a professional image ai user. 
                            Write a single concise english prompt for the text delimited by ```. 
                            Use the same patterns from the example prompts.
                            Your output should only contain the single prompt without further details.
                            ```{text}```
                         """
        self.messages.human_message = HumanMessagePromptTemplate.from_template(human_template)
