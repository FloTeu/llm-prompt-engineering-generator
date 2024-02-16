import pydantic_core
import pytest
import json

import yaml
from langchain.prompts import SystemMessagePromptTemplate

from llm_prompting_gen.models.prompt_engineering import PromptEngineeringMessages, PromptElements


def test_data_class_initialising_json():
    """Test if we can init a PromptEngineeringMessages instance based on some test json files"""
    for file_name in ["sentiment", "kindergartner"]:
        # Test loading json
        pe_messages = PromptEngineeringMessages.from_json(f"tests/test_templates/{file_name}.json")
        # Test creating chat prompt template
        pe_messages.get_chat_prompt_template()

def test_data_class_initialising_yaml():
    """Test if we can init a PromptEngineeringMessages instance based on some test yaml files"""
    for file_name in ["sentiment"]:
        # Test loading yaml
        pe_messages = PromptEngineeringMessages.from_yaml(f"tests/test_templates/{file_name}.yaml")
        # Test creating chat prompt template
        pe_messages.get_chat_prompt_template()

def test_data_class_initialising_json_with_yaml():
    """Test if init with wrong file type raises the expected exception"""
    for file_name in ["sentiment"]:
        # Test loading yaml
        with pytest.raises(AssertionError) as exc:
            PromptEngineeringMessages.from_json(f"tests/test_templates/{file_name}.yaml")
        assert "does not have the correct .json type" in exc.value.args[0]

def test_data_class_initialising_yaml_with_json():
    """Test if init with wrong file type raises the expected exception"""
    for file_name in ["sentiment", "kindergartner"]:
        # Test loading json
        with pytest.raises(AssertionError) as exc:
            PromptEngineeringMessages.from_yaml(f"tests/test_templates/{file_name}.json")
        assert "does not have the correct .yaml type" in exc.value.args[0]

def test_few_shot_string_examples():
    """Test if examples can be provided without human ai interaction"""
    prompt_elements = PromptElements(examples=["positive", "negative", "neutral"])
    prompt_messages = PromptEngineeringMessages.from_pydantic(prompt_elements)
    example_msg = prompt_messages.messages["examples"][0]
    assert type(example_msg) == SystemMessagePromptTemplate
    assert "Example 1: positive" in example_msg.format().content
    assert "Example 2: negative" in example_msg.format().content
    assert "Example 3: neutral" in example_msg.format().content

def test_order_prompt_engineering_messages_by_json():
    """Test whether the order of the original local json file is retained"""
    # test file has a different order compared to PromptElements fields
    file_path = "tests/test_templates/order_test.json"
    prompt_messages = PromptEngineeringMessages.from_json(file_path)
    with open(file_path, "r") as fp:
        message_dict = json.load(fp)
    expected_pe_messages_order = list(message_dict.keys())

    # Test if original json order is retained, after instance creation
    assert expected_pe_messages_order == list(prompt_messages.messages.keys())

def test_order_prompt_engineering_messages_by_json():
    """Test whether the order of the original local json file is retained"""
    # test file has a different order compared to PromptElements fields
    file_path = "tests/test_templates/order_test.json"
    prompt_messages = PromptEngineeringMessages.from_json(file_path)
    with open(file_path, "r") as fp:
        message_dict = json.load(fp)
    expected_pe_messages_order = list(message_dict.keys())

    # Test if original json order is retained, after instance creation
    assert expected_pe_messages_order == list(prompt_messages.messages.keys())

def test_order_prompt_engineering_messages_by_yaml():
    """Test whether the order of the original local yaml file is retained"""
    # test file has a different order compared to PromptElements fields
    file_path = "tests/test_templates/order_test.yaml"
    prompt_messages = PromptEngineeringMessages.from_yaml(file_path)
    with open(file_path, "r") as fp:
        message_dict = yaml.safe_load(fp)
    expected_pe_messages_order = list(message_dict.keys())

    # Test if original json order is retained, after instance creation
    assert expected_pe_messages_order == list(prompt_messages.messages.keys())

def test_big_context_in_yaml():
    """Test whether context in yaml file is correctly transferred to code"""
    file_path = "tests/test_templates/midjourney_prompt_gen_shirt_design_cartoon_style.yaml"
    prompt_messages = PromptEngineeringMessages.from_yaml(file_path)
    expected_context = """Midjourney is an artificial intelligence-powered artwork generator. It explores new thought mediums and expands the human species' imaginative powers. It is a small self-funded team focused on design, human infrastructure, and artificial intelligence. 
Midjourney routinely releases new model versions to improve efficiency, coherency, and quality. The latest model is the default, but other models can be used using the --version or --v parameter or by using the /settings command and selecting a model version. Different models excel at different types of images.
The Midjourney V5 model is the newest and most advanced model, released on March 15th, 2023. To use this model, add the --v 5 parameter to the end of a prompt, or use the /settings command and select 5️⃣ MJ Version 5. This model has very high Coherency, excels at interpreting natural language prompts, is higher resolution, and supports advanced features like repeating patterns with --tile
A Prompt is a short text phrase that the Midjourney Bot interprets to produce an image. The Midjourney Bot breaks down the words and phrases in a prompt into smaller pieces, called tokens, that can be compared to its training data and then used to generate an image. A well-crafted prompt can help make unique and exciting images.
Prompts can be very simple. Single words (or even an emoji!) will produce an image. Very short prompts will rely heavily on Midjourney’s default style, so a more descriptive prompt is better for a unique look. However, super-long prompts aren’t always better. Concentrate on the main concepts you want to create.
The Midjourney Bot does not understand grammar, sentence structure, or words like humans. Word choice also matters. More specific synonyms work better in many circumstances. Instead of big, try gigantic, enormous, or immense. Remove words when possible. Fewer words mean each word has a more powerful influence. Use commas, brackets, and hyphens to help organize your thoughts, but know the Midjourney Bot will not reliably interpret them. The Midjourney Bot does not consider capitalization. Midjourney Model Version 4 is slightly better than other models at interpreting traditional sentence structure.
It is better to describe what you want instead of what you don’t want. If you ask for a party with “no cake,” your image will probably include a cake. If you want to ensure an object is not in the final image, try advance prompting using the --no parameter.
Anything left unsaid may suprise you. Be as specific or vague as you want, but anything you leave out will be randomized. Being vague is a great way to get variety, but you may not get the specific details you want.

Try to be clear about any context or details that are important to you. Think about:

Subject: person, animal, character, location, object, etc.
Medium: photo, painting, illustration, sculpture, doodle, tapestry, etc.
Environment: indoors, outdoors, on the moon, in Narnia, underwater, the Emerald City, etc.
Lighting: soft, ambient, overcast, neon, studio lights, etc
Color: vibrant, muted, bright, monochromatic, colorful, black and white, pastel, etc.
Mood: Sedate, calm, raucous, energetic, etc.
Composition: Portrait, headshot, closeup, birds-eye view, etc.
How to Switch Models
Use the Version or Test Parameter
Add --v 4 --v 5 --v 5.1 --v 5.1 --style raw --v 5.2 --v 5.2 --style raw --niji 5 --niji 5 --style cute --niji 5 --style expressive --niji 5 --style original or --niji 5 --style scenic to the end of your prompt."""

    assert expected_context == prompt_messages.messages["context"].format().content

def test_wrong_format_in_json():
    """Test whether code handles wrong format """
    file_path = "tests/test_templates/wrong_format.json"
    with pytest.raises(pydantic_core.ValidationError) as exc:
        prompt_messages = PromptEngineeringMessages.from_json(file_path)

    assert "role" == exc.value.errors()[0]["loc"][0]
    assert "Input should be a valid string" == exc.value.errors()[0]["msg"]
    assert "instruction" == exc.value.errors()[1]["loc"][0]
    assert "Input should be a valid string" == exc.value.errors()[1]["msg"]
    assert "examples" == exc.value.errors()[2]["loc"][0]
    assert "Input should be a valid list" == exc.value.errors()[2]["msg"]
    assert "examples" == exc.value.errors()[3]["loc"][0]
    assert "Input should be a valid list" == exc.value.errors()[3]["msg"]
