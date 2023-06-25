from langchain.prompts import SystemMessagePromptTemplate

from llm_few_shot_gen.constants import INSTRUCTOR_USER_NAME
from llm_few_shot_gen.generators.prompt import ParsableTextToImagePromptGenerator
from llm_few_shot_gen.context import midjourney as midjourney_context


class MidjourneyPromptGenerator(ParsableTextToImagePromptGenerator):

    def _set_context(self):
        """Extends self.messages with context and type List[SystemMessagePromptTemplate]"""
        context_messages = []
        context_messages.append(SystemMessagePromptTemplate.from_template(
            "Here are some general information about the Midjourney company. ",
            additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(
                midjourney_context.midjourney_company_information,
                additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(
                midjourney_context.midjourney_v5_general_description,
                additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(
                midjourney_context.midjourney_v5_additional_description,
                additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(
            "Now i will provide you some information about prompt engineering for Midjourney.",
            additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(
            SystemMessagePromptTemplate.from_template(midjourney_context.prompt_general_description,
                                                      additional_kwargs={"name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(
            midjourney_context.prompt_length,
            additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(
            midjourney_context.prompt_grammer,
            additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(
            midjourney_context.prompt_what_you_want,
            additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(
            midjourney_context.prompt_details,
            additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        context_messages.append(SystemMessagePromptTemplate.from_template(
            midjourney_context.midjourney_model_switch,
            additional_kwargs={
                                                                           "name": INSTRUCTOR_USER_NAME}))
        self.messages.context = context_messages
