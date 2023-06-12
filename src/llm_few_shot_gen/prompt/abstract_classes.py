import logging
from typing import List

from langchain.base_language import BaseLanguageModel
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import OutputParserException
from langchain.output_parsers import RetryWithErrorOutputParser

from llm_few_shot_gen.abstract_classes import AbstractFewShotGenerator
from llm_few_shot_gen.constants import INSTRUCTOR_USER_NAME
from llm_few_shot_gen.data_classes import FewShotGenerationMessages
from llm_few_shot_gen.prompt.data_classes import ImagePromptOutputModel

class AbstractTextToImagePromptGenerator(AbstractFewShotGenerator):
    """
    Abstract prompt generator class. Subclasses have the ability to generate text-to-image prompts.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.messages: FewShotGenerationMessages = FewShotGenerationMessages()

        # Set up a parser
        self.output_parser = PydanticOutputParser(pydantic_object=ImagePromptOutputModel)

    def _set_system_instruction(self):
        """System message to instruct the llm model how it should act"""
        self.messages.instruction_message = SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant, that helps me create text-to-image prompts.
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
        # human_template = """
        #                     I want you to act as a professional image ai user.
        #                     Write a single concise english prompt for the text delimited by ```.
        #                     Use the same patterns from the example prompts.
        #                     Your output should only contain the single prompt without further details.
        #                     ```{text}```
        #                  """
        human_template = """
                            Complete the following tasks in the right order:
                            1. Try to extract the applied art style of the example prompts that the instructor provided you before.
                            2. Write five concise english prompts with the content "{text}". Your suggestions should include your found styles of step 1 and use the same patterns as the example prompts.
                            {format_instructions}
                        """

        # Inject instructions into the prompt template.
        self.messages.human_message = HumanMessagePromptTemplate.from_template(
            human_template
        )

    def generate(self, *args, **kwargs) -> ImagePromptOutputModel:
        self._fill_messages()
        chat_prompt: ChatPromptTemplate = self.messages.get_chat_prompt_template()
        chat_prompt.partial_variables = {"format_instructions": self.output_parser.get_format_instructions()}
        _input = chat_prompt.format_prompt(*args, **kwargs)
        output = self.llm(_input.to_messages())
        try:
            return self.output_parser.parse(output.content)
        except OutputParserException:
            logging.warning("Could not parse llm output to pydantic class. Retry...")
            retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.output_parser, llm=self.llm)
            return retry_parser.parse_with_prompt(output.content, _input)