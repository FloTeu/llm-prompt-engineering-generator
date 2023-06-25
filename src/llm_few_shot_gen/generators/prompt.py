from typing import List

from pydantic import BaseModel
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.base_language import BaseLanguageModel

from llm_few_shot_gen.constants import INSTRUCTOR_USER_NAME
from llm_few_shot_gen.generators.base import BaseParsableFewShotGenerator
from llm_few_shot_gen.models.output import ImagePromptOutputModel


class ParsableTextToImagePromptGenerator(BaseParsableFewShotGenerator):
    """
    Abstract prompt generator class. Subclasses have the ability to generate text-to-image prompts.
    """

    def __init__(self, llm: BaseLanguageModel, pydantic_cls: BaseModel=ImagePromptOutputModel):
        super().__init__(llm=llm, pydantic_cls=pydantic_cls)

    def _set_system_instruction(self):
        """System message to instruct the llm model how it should act"""
        self.messages.instruction = SystemMessagePromptTemplate.from_template("""
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

    def _set_io_prompt(self):
        """Human message which contains the input for the prompt generation"""
        human_template = """
        Complete the following tasks in the right order:
        1. Try to extract the overarching styles or artists from the example prompts given to you by the instructor. Please only extract them if they appear in at least one example prompt.
        2. Write five concise english prompts with the content "{text}". Your suggestions should include your found styles or artists of step 1 and use the same patterns as the example prompts.
        """

        # Inject instructions into the prompt template.
        self.messages.io_prompt = HumanMessagePromptTemplate.from_template(
            human_template
        )

    def generate(self, text) -> ImagePromptOutputModel:
        return super().generate(text=text)
