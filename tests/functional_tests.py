from langchain.chat_models import ChatOpenAI
from llm_few_shot_gen.generators import FewShotGenerator, ParsableFewShotGenerator
from llm_few_shot_gen.models.output import ImagePromptOutputModel
from llm_few_shot_gen.models.prompt_engineering import PEMessages


def test_data_class_initialising():
    """Test if we can init a templates class based on some test json files"""
    for file_name in ["sentiment", "kindergartner"]:
        # Test loading json
        pe_messages = PEMessages.from_json(f"templates/{file_name}.json")
        # Test creating chat prompt template
        pe_messages.get_chat_prompt_template()


def test_sentiment_generator():
    """Test if basic prompt generator class can be initialised an executed"""
    llm = ChatOpenAI(temperature=0.0)
    prompt_generator = FewShotGenerator.from_json(f"templates/sentiment.json", llm)
    sentiment = prompt_generator.generate(text="My dog looks so cute today")
    assert sentiment == "positive"

def test_parsed_midjourney_prompt_generator():
    """Test if basic prompt generator class can be initialised an executed"""
    llm = ChatOpenAI(temperature=0.0)
    prompt_generator = ParsableFewShotGenerator.from_json(f"templates/midjourney_prompt_gen_shirt_design_cartoon_style.json", llm=llm, pydantic_cls=ImagePromptOutputModel)
    prompt_generator.prompt_elements.input = """
Complete the following tasks in the right order:
1. Try to extract the overarching styles or artists from the example prompts given to you by the instructor. Please only extract them if they appear in at least one example prompt.
2. Write five concise english prompts with the content "{text}". Your suggestions should include your found styles or artists of step 1 and use the same patterns as the example prompts.
    """
    llm_parsed_output: ImagePromptOutputModel = prompt_generator.generate(text="dog")
    assert type(llm_parsed_output) == ImagePromptOutputModel
    assert "cartoon" in llm_parsed_output.few_shot_styles

