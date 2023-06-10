from langchain.chat_models import ChatOpenAI

def get_chat_gpt_model(temperature=0.7) -> ChatOpenAI:
    return ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo")
