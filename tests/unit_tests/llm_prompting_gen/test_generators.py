import json
from unittest.mock import patch

import yaml
from langchain.chains.llm import LLMChain
from langchain.output_parsers import RetryWithErrorOutputParser
from pydantic import BaseModel
from llm_prompting_gen.generators import PromptEngineeringGenerator, ParsablePromptEngineeringGenerator
from llm_prompting_gen.models.prompt_engineering import PromptElements

def test_extra_field(test_llm):
    """
    PromptElements can contain extra fields.
    This fields are supposed to be system messages in the final prompt
    """
    pe_gen = PromptEngineeringGenerator.from_json(f"tests/test_templates/sentiment.json", llm=test_llm)
    assert "examples_intro" in pe_gen.message_order, "Extra field 'examples_intro' is not included in message_order attribute"
    assert "examples_intro" in pe_gen.prompt_elements.model_extra.keys(), "Extra field 'examples_intro' is not included in PromptELements"
    # test if xtra field is included in final prompt
    chat_prompt_template = pe_gen._get_messages().get_chat_prompt_template(message_order=pe_gen.message_order)
    assert pe_gen.prompt_elements.examples_intro in chat_prompt_template.format(text=""), "Extra field 'examples_intro' is not included in final prompt"


def test_message_order(test_llm):
    """
    Message order can either be defined implicitly by json
    or explicitly by class initialization.
    Both cases are tested here.
    """
    # Test implicit order by json
    # output_format is per default between context and examples. In case of sentiment json its the last element
    pe_gen = PromptEngineeringGenerator.from_json(f"tests/test_templates/sentiment.json", llm=test_llm)
    assert "output_format" in pe_gen.message_order[-1], "'output_format' is supposed to be at the end of the prompt element order"
    # Test if last element of chat prompt template is also output_format
    chat_prompt_template = pe_gen._get_messages().get_chat_prompt_template(message_order=pe_gen.message_order)
    assert pe_gen.prompt_elements.output_format == chat_prompt_template.messages[-1].prompt.partial_variables["format_instructions"]

    # Test explicit order by class initialization
    prompt_elements = PromptElements(role="Sentiment classifier", examples=["positive", "negative", "neutral"])
    pe_gen = PromptEngineeringGenerator(llm=test_llm, prompt_elements=prompt_elements, message_order=["examples", "role"])
    assert "role" in pe_gen.message_order[
        -1], "'role' is supposed to be at the end of the prompt element order"
    # Test if last element of chat prompt template is also output_format
    chat_prompt_template = pe_gen._get_messages().get_chat_prompt_template(message_order=pe_gen.message_order)
    assert "Sentiment classifier" == chat_prompt_template.messages[-1].prompt.template

def test_from_json(test_llm):
    """Test if we can init a prompt engineering gen instance based on some test json files"""
    class TestOutputClass(BaseModel):
        output: str

    for file_name in ["sentiment", "kindergartner", "order_test", "midjourney_prompt_gen_shirt_design_cartoon_style"]:
        file_path = f"tests/test_templates/{file_name}.json"
        with open(file_path, "r") as fp:
            expected_message_keys = set(json.load(fp).keys())

        # Test loading json
        pe_gen = PromptEngineeringGenerator.from_json(file_path, llm=test_llm)
        assert len(expected_message_keys - set(pe_gen._get_messages().messages.keys())) == 0
        pe_gen = ParsablePromptEngineeringGenerator.from_json(file_path, llm=test_llm, pydantic_cls=TestOutputClass)
        assert len(expected_message_keys - set(pe_gen._get_messages().messages.keys())) == 0


def test_from_yaml(test_llm):
    """Test if we can init a prompt engineering gen instance based on some test yaml files"""
    class TestOutputClass(BaseModel):
        output: str

    for file_name in ["sentiment", "order_test", "midjourney_prompt_gen_shirt_design_cartoon_style"]:
        file_path = f"tests/test_templates/{file_name}.yaml"
        with open(file_path, "r") as fp:
            expected_message_keys = set(yaml.safe_load(fp).keys())

        # Test loading yaml
        pe_gen = PromptEngineeringGenerator.from_yaml(file_path, llm=test_llm)
        assert len(expected_message_keys - set(pe_gen._get_messages().messages.keys())) == 0
        pe_gen = ParsablePromptEngineeringGenerator.from_yaml(file_path, llm=test_llm, pydantic_cls=TestOutputClass)
        assert len(expected_message_keys - set(pe_gen._get_messages().messages.keys())) == 0

def test_parsable_generate(test_llm):
    """Test if generate with retry functionality works as expected"""
    class TestOutputClass(BaseModel):
        output: str

    expected_output = TestOutputClass(output="test")

    with patch.object(LLMChain, '_call') as mock_get_data:
        unparsable_output = {"text": '{"out": "wrong key"}'}
        mock_get_data.return_value = unparsable_output
        prompt_elements = PromptElements(instruction="Create a example object of the provided output format")
        pe_gen = ParsablePromptEngineeringGenerator(prompt_elements=prompt_elements, llm=test_llm, pydantic_cls=TestOutputClass)
        with patch.object(RetryWithErrorOutputParser, "parse_with_prompt") as mock_output_parser:
            mock_output_parser.return_value = expected_output
            gen_output = pe_gen.generate()

    assert gen_output == expected_output

async def test_parsable_agenerate(test_llm):
    """Test if asynchronous generate with retry functionality works as expected"""
    class TestOutputClass(BaseModel):
        output: str

    expected_output = TestOutputClass(output="test")

    with patch.object(LLMChain, '_acall') as mock_get_data:
        unparsable_output = {"text": '{"out": "wrong key"}'}
        mock_get_data.return_value = unparsable_output
        prompt_elements = PromptElements(instruction="Create a example object of the provided output format")
        pe_gen = ParsablePromptEngineeringGenerator(prompt_elements=prompt_elements, llm=test_llm, pydantic_cls=TestOutputClass)
        with patch.object(RetryWithErrorOutputParser, "aparse_with_prompt") as mock_output_parser:
            mock_output_parser.return_value = expected_output
            gen_output = await pe_gen.agenerate()

    assert gen_output == expected_output

