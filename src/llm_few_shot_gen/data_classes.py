from typing import List, Optional
from dataclasses import dataclass
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate

@dataclass
class FewShotGenerationMessages:
    instruction_message: Optional[SystemMessagePromptTemplate] = None
    context: Optional[List[SystemMessagePromptTemplate]] = None
    few_shot_examples: Optional[List[SystemMessagePromptTemplate]] = None
    human_message: Optional[HumanMessagePromptTemplate] = None

    def is_instruction_known(self):
        return bool(self.instruction_message)

    def is_context_known(self):
        return bool(self.context)

    def are_few_shot_examples_set(self):
        return bool(self.few_shot_examples)

    def is_human_message_set(self):
        return bool(self.human_message)

    def get_chat_prompt_template(self) -> ChatPromptTemplate:
        assert self.instruction_message != None
        assert self.context != None
        assert self.few_shot_examples != None
        assert self.human_message != None
        messages = [
            self.instruction_message,
            *self.context,
            *self.few_shot_examples,
            self.human_message
        ]
        return ChatPromptTemplate.from_messages(messages)
