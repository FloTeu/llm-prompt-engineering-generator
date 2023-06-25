from typing import List, Optional
from dataclasses import dataclass
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

@dataclass
class FewShotGenerationMessages:
    """
    Dataclass which contains everything to create a chat dialog
    for generating a new output based on few shot examples
    """
    instruction: Optional[SystemMessagePromptTemplate] = None # Instruction for LLM model
    context: Optional[List[SystemMessagePromptTemplate]] = None # Context of the Few Shot learning use case
    few_shot_examples: Optional[List[SystemMessagePromptTemplate]] = None # Few Shot Examples
    io_prompt: Optional[HumanMessagePromptTemplate] = None # Output instruction, which might contain some human input as well. I/O=input/output

    def get_chat_prompt_template(self) -> ChatPromptTemplate:
        assert self.instruction != None
        assert self.context != None
        assert self.few_shot_examples != None
        assert self.io_prompt != None
        messages = [
            self.instruction,
            *self.context,
            *self.few_shot_examples,
            self.io_prompt
        ]
        return ChatPromptTemplate.from_messages(messages)
