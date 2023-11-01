from langchain.chat_models import ChatOpenAI
from llm_prompting_gen.generators import PromptEngineeringGenerator, ParsablePromptEngineeringGenerator
from llm_prompting_gen.models.output import ImagePromptOutputModel
from llm_prompting_gen.models.prompt_engineering import PEMessages


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
    prompt_generator = PromptEngineeringGenerator.from_json(f"templates/sentiment.json", llm)
    sentiment = prompt_generator.generate(text="My dog looks so cute today")
    assert sentiment == "positive"

def test_keyword_extractor_generator():
    """Test if output format is correct with keyword extractor template"""
    llm = ChatOpenAI(temperature=0.0)
    prompt_generator = PromptEngineeringGenerator.from_json(f"templates/keyword_extractor.json", llm=llm)
    # Wiki article about gen AI
    llm_output = prompt_generator.generate(text="""
Generative artificial intelligence (also generative AI or GenAI[1]) is artificial intelligence capable of generating text, images, or other media, using generative models.[2][3][4] Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.
In the early 2020s, advances in transformer-based deep neural networks enabled a number of generative AI systems notable for accepting natural language prompts as input. These include large language model chatbots such as ChatGPT, Bing Chat, Bard, and LLaMA, and text-to-image artificial intelligence art systems such as Stable Diffusion, Midjourney, and DALL-E.
    """)
    # test if we can transform the output into an list
    assert len(llm_output.split(",")) > 10, "LLM could not extract more than 5 keywords in comma separated format"

def test_parsed_midjourney_prompt_generator():
    """Test if basic prompt generator class can be initialised an executed"""
    llm = ChatOpenAI(temperature=0.0)
    prompt_generator = ParsablePromptEngineeringGenerator.from_json(f"templates/midjourney_prompt_gen_shirt_design_cartoon_style.json", llm=llm, pydantic_cls=ImagePromptOutputModel)
    prompt_generator.prompt_elements.input = """
Complete the following tasks in the right order:
1. Try to extract the overarching styles or artists from the example prompts given to you by the instructor. Please only extract them if they appear in at least one example prompt.
2. Write five concise english prompts with the content "{text}". Your suggestions should include your found styles or artists of step 1 and use the same patterns as the example prompts.
    """
    llm_parsed_output: ImagePromptOutputModel = prompt_generator.generate(text="dog")
    assert type(llm_parsed_output) == ImagePromptOutputModel
    assert "cartoon" in llm_parsed_output.few_shot_styles or "cartoonish" in llm_parsed_output.few_shot_styles


