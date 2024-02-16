import pytest
from langchain.chat_models import ChatOpenAI

@pytest.fixture
def test_llm() -> ChatOpenAI:
    llm = ChatOpenAI(temperature=0.0, openai_api_key="test")
    return llm
