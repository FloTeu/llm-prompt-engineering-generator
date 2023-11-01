from typing import Optional, List

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, \
    FewShotChatMessagePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from llm_prompting_gen.constants import INSTRUCTOR_USER_NAME

class PEFewShotExample(BaseModel):
    human: Optional[str] = Field(description="Example human message.", examples=["What is 2 + 3?"])
    ai: str = Field(description="Example ai message as answer to the human message. Or standalone example without human message.", examples=["5"])


class PEFewShotExamples(BaseModel):
    intro: Optional[str] = Field(description="If set, the few shot examples are introduced")
    human_ai_interaction: List[PEFewShotExample] = Field(
        description="List of example human ai interaction. Shows the LLM how the output should look like.")


class PromptElements(BaseModel):
    """
    Defines all elements of a prompt
    """
    role: Optional[str] = Field(description="The role in which the LLM should respond")
    instruction: Optional[str] = Field(description="The task of the LLM")
    context: Optional[str] = Field(description="Context with relevant information to solve the task")
    output_format: Optional[str] = Field(description="Description how the LLM output format should look like")
    examples: Optional[PEFewShotExamples] = Field(
        description="Few shot examples with optional introduction")
    input: Optional[str] = Field(
        description="Target which the LLM should execute the task on. Could be for example a user question, or a text block to summarize.")

    def is_any_set(self) -> bool:
        """
        Whether at least one prompt element is set.
        Otherwise, an LLM cannot handle prompt input and therefore False is returned
        """
        return any([self.role, self.instruction, self.context, self.examples, self.input, self.output_format])

    def get_few_shot_intro_prompt_template(self) -> Optional[SystemMessagePromptTemplate]:
        if self.examples and self.examples.intro:
            return SystemMessagePromptTemplate.from_template(self.examples.intro, additional_kwargs={"name": INSTRUCTOR_USER_NAME})

    def get_few_shot_chat_msg_prompt_template(self) -> FewShotChatMessagePromptTemplate:
        """Returns langchain few shot example prompt template"""
        assert self.examples, "examples are not set yet"
        examples = []
        is_only_ai_interaction = all([not example.human for example in self.examples.human_ai_interaction])
        for example in self.examples.human_ai_interaction:
            if is_only_ai_interaction:
                examples.append({"output": example.ai})
            else:
                examples.append({"input": example.human, "output": example.ai})
        if is_only_ai_interaction:
            example_prompt = ChatPromptTemplate.from_messages(
                [
                    ("ai", "{output}")
                ]
            )
        else:
            example_prompt = ChatPromptTemplate.from_messages(
                [
                    ("human", "{input}"),
                    ("ai", "{output}"),
                ]
            )
        return FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )


class PEMessages(BaseModel):
    """
    Dataclass which contains everything to create a LLM Output
    based on prompt engineering techniques.
    Transforms strings into langchain prompt template messages.
    """
    role: Optional[SystemMessagePromptTemplate] = Field(description="The role in which the LLM should respond")
    instruction: Optional[SystemMessagePromptTemplate] = Field(description="The task of the LLM")
    context: Optional[SystemMessagePromptTemplate] = Field(
        description="Context with relevant information to solve the task")
    output_format: Optional[SystemMessagePromptTemplate] = Field(
        description="Description how the LLM output format should look like")
    examples_intro: Optional[SystemMessagePromptTemplate] = Field(
        description="Optional intro block to explain LLM how to handle following examples")
    examples: Optional[FewShotChatMessagePromptTemplate] = Field(
        description="List of (few-shot) examples, how the output should look like")
    input: Optional[HumanMessagePromptTemplate] = Field(
        description="Target which the LLM should execute the task on. Could be for example a user question, or a text block to summarize.")

    @classmethod
    def from_pydantic(cls, pe_elements: PromptElements):
        # Create output format system message
        output_format_msg = None
        if pe_elements.output_format:
            output_format_prompt = PromptTemplate(
                template="{format_instructions}",
                input_variables=[],
                partial_variables={"format_instructions": pe_elements.output_format}
            )
            output_format_msg = SystemMessagePromptTemplate(prompt=output_format_prompt)
        examples_intro_msg = pe_elements.get_few_shot_intro_prompt_template()

        return cls(
            role=SystemMessagePromptTemplate.from_template(pe_elements.role, additional_kwargs={"name": INSTRUCTOR_USER_NAME}) if pe_elements.role else None,
            instruction=SystemMessagePromptTemplate.from_template(
                pe_elements.instruction, additional_kwargs={"name": INSTRUCTOR_USER_NAME}) if pe_elements.instruction else None,
            context=SystemMessagePromptTemplate.from_template(pe_elements.context, additional_kwargs={"name": INSTRUCTOR_USER_NAME}) if pe_elements.context else None,
            examples_intro=examples_intro_msg,
            examples=pe_elements.get_few_shot_chat_msg_prompt_template() if pe_elements.examples else None,
            input=HumanMessagePromptTemplate.from_template(pe_elements.input) if pe_elements.input else None,
            output_format=output_format_msg
        )

    @classmethod
    def from_json(cls, file_path):
        return cls.from_pydantic(PromptElements.parse_file(file_path))

    @classmethod
    def from_yaml(cls, file_path):
        # TODO
        raise NotImplementedError

    def get_chat_prompt_template(self) -> ChatPromptTemplate:
        """Combines all prompt element messages into one chat prompt template"""
        messages = []
        if self.role:
            messages.append(self.role)
        if self.instruction:
            messages.append(self.instruction)
        if self.context:
            messages.append(self.context)
        if self.output_format:
            messages.append(self.output_format)
        if self.examples_intro:
            messages.append(self.examples_intro)
        if self.examples:
            messages.append(self.examples)
        if self.input:
            messages.append(self.input)

        return ChatPromptTemplate.from_messages(messages)
