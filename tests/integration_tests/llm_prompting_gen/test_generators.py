from typing import List
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from llm_prompting_gen.generators import PromptEngineeringGenerator, ParsablePromptEngineeringGenerator
from llm_prompting_gen.models.prompt_engineering import PromptElements
from llm_prompting_gen.models.output import KeywordExtractorOutput


def test_sentiment_generator():
    """Test if basic prompt generator class can be initialised an executed"""
    llm = ChatOpenAI(temperature=0.0)
    sentiment_gen = PromptEngineeringGenerator.from_json(f"tests/test_templates/sentiment.json", llm)
    sentiment = sentiment_gen.generate(text="My dog looks so cute today")
    assert sentiment == "positive"


def test_parsed_keyword_extractor():
    """Tests if we can extract keywords and parse output to pydantic"""
    llm = ChatOpenAI(temperature=0.0)
    text = """
    Generative artificial intelligence (also generative AI or GenAI[1]) is artificial intelligence capable of generating text, images, or other media, using generative models.[2][3][4] Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.
    In the early 2020s, advances in transformer-based deep neural networks enabled a number of generative AI systems notable for accepting natural language prompts as input. These include large language model chatbots such as ChatGPT, Bing Chat, Bard, and LLaMA, and text-to-image artificial intelligence art systems such as Stable Diffusion, Midjourney, and DALL-E.
    """
    keyword_gen = ParsablePromptEngineeringGenerator.from_json(f"tests/test_templates/keyword_extractor.json", llm, pydantic_cls=KeywordExtractorOutput)
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
    prompt_generator = ParsablePromptEngineeringGenerator.from_json(f"tests/test_templates/midjourney_prompt_gen_shirt_design_cartoon_style.json", llm=llm, pydantic_cls=ImagePromptOutputModel)
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
    keyword_extractor = PromptEngineeringGenerator.from_json("tests/test_templates/keyword_extractor.json", llm=llm)
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


