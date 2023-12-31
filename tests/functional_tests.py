from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from pydantic import BaseModel, Field
from llm_prompting_gen.generators import PromptEngineeringGenerator, ParsablePromptEngineeringGenerator
from llm_prompting_gen.models.prompt_engineering import PromptEngineeringMessages, PromptElements
from llm_prompting_gen.models.output import KeywordExtractorOutput

def test_data_class_initialising():
    """Test if we can init a templates class based on some test json files"""
    for file_name in ["sentiment", "kindergartner"]:
        # Test loading json
        pe_messages = PromptEngineeringMessages.from_json(f"templates/{file_name}.json")
        # Test creating chat prompt template
        pe_messages.get_chat_prompt_template()

def test_extra_field():
    """
    PromptElements can contain extra fields.
    This fields are supposed to be system messages in the final prompt
    """
    llm = ChatOpenAI(temperature=0.0)
    pe_gen = PromptEngineeringGenerator.from_json(f"templates/sentiment.json", llm=llm)
    assert "examples_intro" in pe_gen.message_order, "Extra field 'examples_intro' is not included in message_order attribute"
    assert "examples_intro" in pe_gen.prompt_elements.model_extra.keys(), "Extra field 'examples_intro' is not included in PromptELements"
    # test if xtra field is included in final prompt
    chat_prompt_template = pe_gen._get_messages().get_chat_prompt_template(message_order=pe_gen.message_order)
    assert pe_gen.prompt_elements.examples_intro in chat_prompt_template.format(text=""), "Extra field 'examples_intro' is not included in final prompt"

def test_message_order():
    """
    Message order can either be defined implicitly by json
    or explicitly by class initialization.
    Both cases are tested here.
    """
    # Test implicit order by json
    # output_format is per default between context and examples. In case of sentiment json its the last element
    llm = ChatOpenAI(temperature=0.0)
    pe_gen = PromptEngineeringGenerator.from_json(f"templates/sentiment.json", llm=llm)
    assert "output_format" in pe_gen.message_order[-1], "'output_format' is supposed to be at the end of the prompt element order"
    # Test if last element of chat prompt template is also output_format
    chat_prompt_template = pe_gen._get_messages().get_chat_prompt_template(message_order=pe_gen.message_order)
    assert pe_gen.prompt_elements.output_format == chat_prompt_template.messages[-1].prompt.partial_variables["format_instructions"]

    # Test explicit order by class initialization
    prompt_elements = PromptElements(role="Sentiment classifier", examples=["positive", "negative", "neutral"])
    pe_gen = PromptEngineeringGenerator(llm=llm, prompt_elements=prompt_elements, message_order=["examples", "role"])
    assert "role" in pe_gen.message_order[
        -1], "'role' is supposed to be at the end of the prompt element order"
    # Test if last element of chat prompt template is also output_format
    chat_prompt_template = pe_gen._get_messages().get_chat_prompt_template(message_order=pe_gen.message_order)
    assert "Sentiment classifier" == chat_prompt_template.messages[-1].prompt.template


def test_sentiment_generator():
    """Test if basic prompt generator class can be initialised an executed"""
    llm = ChatOpenAI(temperature=0.0)
    sentiment_gen = PromptEngineeringGenerator.from_json(f"templates/sentiment.json", llm)
    sentiment = sentiment_gen.generate(text="My dog looks so cute today")
    assert sentiment == "positive"

def test_few_shot_string_examples():
    """Test if examples can be provided without human ai interaction"""
    prompt_elements = PromptElements(examples=["positive", "negative", "neutral"])
    prompt_messages = PromptEngineeringMessages.from_pydantic(prompt_elements)
    example_msg = prompt_messages.messages["examples"][0]
    assert type(example_msg) == SystemMessagePromptTemplate
    assert "Example 1: positive" in example_msg.format().content
    assert "Example 2: negative" in example_msg.format().content
    assert "Example 3: neutral" in example_msg.format().content

def test_parsed_keyword_extractor():
    """Tests if we can extract keywords and parse output to pydantic"""
    llm = ChatOpenAI(temperature=0.0)
    text = """
    Generative artificial intelligence (also generative AI or GenAI[1]) is artificial intelligence capable of generating text, images, or other media, using generative models.[2][3][4] Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.
    In the early 2020s, advances in transformer-based deep neural networks enabled a number of generative AI systems notable for accepting natural language prompts as input. These include large language model chatbots such as ChatGPT, Bing Chat, Bard, and LLaMA, and text-to-image artificial intelligence art systems such as Stable Diffusion, Midjourney, and DALL-E.
    """
    keyword_gen = ParsablePromptEngineeringGenerator.from_json(f"templates/keyword_extractor.json", llm, pydantic_cls=KeywordExtractorOutput)
    keyword_output: KeywordExtractorOutput = keyword_gen.generate(text=text)
    assert "ChatGPT" in keyword_output.keywords


def test_parsed_midjourney_prompt_generator():
    """Test if basic prompt generator class can be initialised an executed"""

    class ImagePromptOutputModel(BaseModel):
        """LLM output format of image prompt generator"""
        few_shot_styles: List[str] = Field(description="Styles existing in the example prompts")
        few_shot_artists: List[str] = Field(description="Artists existing in the example prompts")
        image_prompts: List[str] = Field(description="List of text-to-image prompts")

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

def test_readme_examples():
    """Test the simple examples of the readme docu"""
    # Simply load a JSON file following the format of llm_prompting_gen.models.prompt_engineering.PromptElements
    from llm_prompting_gen.generators import PromptEngineeringGenerator
    # Make sure env variable OPENAI_API_KEY is set
    llm = ChatOpenAI(temperature=0.0)
    keyword_extractor = PromptEngineeringGenerator.from_json("templates/keyword_extractor.json", llm=llm)
    # Wiki article about gen AI
    llm_output = keyword_extractor.generate(text="""
Generative artificial intelligence (also generative AI or GenAI[1]) is artificial intelligence capable of generating text, images, or other media, using generative models.[2][3][4] Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.
In the early 2020s, advances in transformer-based deep neural networks enabled a number of generative AI systems notable for accepting natural language prompts as input. These include large language model chatbots such as ChatGPT, Bing Chat, Bard, and LLaMA, and text-to-image artificial intelligence art systems such as Stable Diffusion, Midjourney, and DALL-E.
""")
    # test if we can transform the output into an list
    assert len(llm_output.split(",")) > 10, "LLM could not extract more than 5 keywords in comma separated format"

    # Or simply create a Prompt Engineering class yourself
    from llm_prompting_gen.models.prompt_engineering import PromptElements
    prompt_elements = PromptElements(role="You are a keyword extractor", instruction="Extract the keyword from the text felimited by '''", input="'''{text}'''")
    llm = ChatOpenAI(temperature=0.0)
    keyword_extractor = PromptEngineeringGenerator(llm=llm, prompt_elements=prompt_elements)
    llm_output = keyword_extractor.generate(text="""
    Generative artificial intelligence (also generative AI or GenAI[1]) is artificial intelligence capable of generating text, images, or other media, using generative models.[2][3][4] Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.
    In the early 2020s, advances in transformer-based deep neural networks enabled a number of generative AI systems notable for accepting natural language prompts as input. These include large language model chatbots such as ChatGPT, Bing Chat, Bard, and LLaMA, and text-to-image artificial intelligence art systems such as Stable Diffusion, Midjourney, and DALL-E.
    """)


