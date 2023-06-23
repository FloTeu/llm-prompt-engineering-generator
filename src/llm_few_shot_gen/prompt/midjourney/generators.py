from langchain.base_language import BaseLanguageModel
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

from llm_few_shot_gen.prompt.midjourney import context as midjourney_context
from llm_few_shot_gen.prompt.abstract_classes import AbstractTextToImagePromptGenerator
from llm_few_shot_gen.constants import INSTRUCTOR_USER_NAME


class MidjourneyPromptGenerator(AbstractTextToImagePromptGenerator):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm)

    def set_context(self):
        context_messages = []
        context_messages.append(SystemMessagePromptTemplate.from_template(
            "Here is some general information about the Midjourney company. ",
            additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(midjourney_context.midjourney_company_information,
                                                      additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(midjourney_context.midjourney_v5_general_description,
                                                      additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(midjourney_context.midjourney_v5_additional_description,
                                                      additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(
            "Now i will provide you some information about prompt engineering for Midjourney.",
            additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(midjourney_context.prompt_general_description,
                                                      additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(midjourney_context.prompt_length,
                                                                       additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(midjourney_context.prompt_grammer,
                                                                       additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(midjourney_context.prompt_what_you_want,
                                                                       additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(midjourney_context.prompt_details,
                                                                       additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(midjourney_context.midjourney_model_switch,
                                                                          additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        self.messages.context = context_messages

