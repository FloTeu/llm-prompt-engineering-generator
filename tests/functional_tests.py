from langchain.chat_models import ChatOpenAI
from llm_few_shot_gen.generators import FewShotGenerator
from llm_few_shot_gen.models.output import ImagePromptOutputModel
from llm_few_shot_gen.models.prompt_engineering import PromptEngineeringMessages


def test_data_class_initialising():
    """Test if we can init a data class based on some test json files"""
    for file_name in ["sentiment", "kindergartner"]:
        # Test loading json
        pe_messages = PromptEngineeringMessages.from_json(f"data/{file_name}.json")
        # Test creating chat prompt template
        pe_messages.get_chat_prompt_template()


def test_prompt_generator():
    """Test if basic prompt generator class can be initialised an executed"""
    llm = ChatOpenAI(temperature=0.0)
    prompt_generator = FewShotGenerator.from_json(f"data/sentiment.json", llm)
    sentiment = prompt_generator.generate(text="My dog looks so cute today")
    assert sentiment == "positive"