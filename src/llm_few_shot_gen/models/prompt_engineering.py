from typing import Optional, List

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, \
    FewShotChatMessagePromptTemplate, ChatMessagePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

class PromptEngineeringFewShotExample(BaseModel):
    human: str = Field(description="Example human message.", examples=["What is 2 + 3?"])
    ai: str = Field(description="Example ai message as answer to the human message.", examples=["5"])


class PromptEngineeringElements(BaseModel):
    """
    Defines all elements of a prompt
    """
    role: Optional[str] = Field(description="The role in which the LLM should respond")
    instruction: Optional[str] = Field(description="The task of the LLM")
    context: Optional[str] = Field(description="Context with relevant information to solve the task")
    examples: Optional[List[PromptEngineeringFewShotExample]] = Field(
        description="List of examples, how the output should look like")
    input: Optional[str] = Field(
        description="Target which the LLM should execute the task on. Could be for example a user question, or a text block to summarize.")
    output_format: Optional[str] = Field(description="Description how the LLM output format should look like")

    def is_any_set(self) -> bool:
        """
        Whether at least one prompt element is set.
        Otherwise, an LLM cannot handle prompt input and therefore False is returned
        """
        return any([self.role, self.instruction, self.context, self.examples, self.input, self.output_format])


    def get_few_shot_chat_msg_prompt_template(self) -> FewShotChatMessagePromptTemplate:
        """Returns langchain few shot example prompt template"""
        examples = []
        for example in self.examples:
            examples.append({"input": example.human, "output": example.ai})
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        return FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

class PromptEngineeringMessages(BaseModel):
    """
    Dataclass which contains everything to create a LLM Output
    based on prompt engineering techniques.
    Transforms strings into langchain prompt template messages.
    """
    role: Optional[SystemMessagePromptTemplate] = Field(description="The role in which the LLM should respond")
    instruction: Optional[SystemMessagePromptTemplate] = Field(description="The task of the LLM")
    context: Optional[SystemMessagePromptTemplate] = Field(
        description="Context with relevant information to solve the task")
    examples: Optional[FewShotChatMessagePromptTemplate] = Field(
        description="List of (few-shot) examples, how the output should look like")
    input: Optional[HumanMessagePromptTemplate] = Field(
        description="Target which the LLM should execute the task on. Could be for example a user question, or a text block to summarize.")
    output_format: Optional[SystemMessagePromptTemplate] = Field(
        description="Description how the LLM output format should look like")


    @classmethod
    def from_pydantic(cls, pe_elements: PromptEngineeringElements):
        return cls(
            role=SystemMessagePromptTemplate.from_template(pe_elements.role) if pe_elements.role else None,
            instruction=SystemMessagePromptTemplate.from_template(pe_elements.instruction) if pe_elements.instruction else None,
            context=SystemMessagePromptTemplate.from_template(pe_elements.context) if pe_elements.context else None,
            examples=pe_elements.get_few_shot_chat_msg_prompt_template() if pe_elements.examples else None,
            input=HumanMessagePromptTemplate.from_template(pe_elements.input) if pe_elements.input else None,
            output_format=SystemMessagePromptTemplate.from_template(pe_elements.output_format) if pe_elements.output_format else None,
        )

    @classmethod
    def from_json(cls, file_path):
        return cls.from_pydantic(PromptEngineeringElements.parse_file(file_path))

    @classmethod
    def from_yaml(cls, file_path):
        #TODO
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
        if self.examples:
            messages.append(self.examples)
        if self.input:
            messages.append(self.input)
        if self.output_format:
            messages.append(self.output_format)

        return ChatPromptTemplate.from_messages(messages)
