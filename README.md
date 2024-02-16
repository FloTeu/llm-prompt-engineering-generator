# LLM Prompt Engineering Generator

[![PyPI - Version](https://img.shields.io/pypi/v/llm-prompting-gen.svg)](https://pypi.org/project/llm-prompting-gen)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm-prompting-gen.svg)](https://pypi.org/project/llm-prompting-gen)

-----


## Installation

```console
pip install llm-prompting-gen
```

## What is LLM Prompt Engineering Generator?
Large language models (LLMs) are emerging as a transformative technology, enabling developers to build applications that they previously could not. However, fine-tune them is not always possible or too expansive. Prompt Engineering techniques like In-Context Learning or Few-Shot Learning tries to solve this problem. Based on `langchain`, this library enables prompt engineering in a straightforward way. 

## How can I use it?
This library consists of two main building blocks LLMs and Prompt Engineering. LLMs are provided by [langchain](https://github.com/langchain-ai/langchain), whereas the Prompt Engineering is part of this library. 
Both building blocks are combined in the `generators` module. The class `PromptEngineeringGenerator` defines all requirements for a LLM to generate text based on prompt engineering techniques. If you want your output to be parsed into an pydantic dataclass checkout the class `ParsablePromptEngineeringGenerator`. 

### Simple Example
```python
# Task: extract keywords
# Wiki article about gen AI
text_with_keywords = """
Generative artificial intelligence (also generative AI or GenAI[1]) is artificial intelligence capable of generating text, images, or other media, using generative models.[2][3][4] Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.
In the early 2020s, advances in transformer-based deep neural networks enabled a number of generative AI systems notable for accepting natural language prompts as input. These include large language model chatbots such as ChatGPT, Bing Chat, Bard, and LLaMA, and text-to-image artificial intelligence art systems such as Stable Diffusion, Midjourney, and DALL-E.
"""
from llm_prompting_gen.generators import PromptEngineeringGenerator
from llm_prompting_gen.models.prompt_engineering import PromptElements
from langchain.chat_models import ChatOpenAI

## Option 1: Simply load a JSON or YAML file following the format of llm_prompting_gen.models.prompt_engineering.PromptElements
# Make sure env variable OPENAI_API_KEY is set
llm = ChatOpenAI(temperature=0.0)
# JSON 
keyword_extractor = PromptEngineeringGenerator.from_json("templates/keyword_extractor.json", llm=llm)
# YAML
keyword_extractor = PromptEngineeringGenerator.from_yaml("templates/keyword_extractor.yaml", llm=llm)
llm_output = keyword_extractor.generate(text=text_with_keywords)

## Option 2: Simply create a Prompt Engineering class yourself
prompt_elements = PromptElements(role="You are a keyword extractor", instruction="Extract the keyword from the text delimited by '''", input="'''{text}'''")
llm = ChatOpenAI(temperature=0.0)
keyword_extractor = PromptEngineeringGenerator(llm=llm, prompt_elements=prompt_elements)
llm_output = keyword_extractor.generate(text=text_with_keywords)
```

## How to customise the class for my own use case?
The class `PromptEngineeringGenerator` contains two core parts 1. LLM 2. prompt engineering dataclass. If you want to initialise the generator class for your custom use case, you need to define a prompt engineering JSON file matching the format of `llm_prompting_gen.models.prompt_engineering.PromptElements`.

The JSON file can contain the following prompt elements in any combination tailored to your use case:

**Role**:
The role in which the LLM should respond

**Instruction**:
The task of the LLM

**Context**:
Context with relevant information to solve the task

**Output Format**:
Description how the LLM output format should look like

**Few Shot Examples**:
Few shot examples with optional introduction

**Input**:
Target which the LLM should execute the task on. Could be for example a user question, or a text block to summarize.


### Showcases
* [Notebook Showcase: Midjourney Prompt](https://github.com/FloTeu/llm-prompt-engineering-generator/blob/main/notebooks/few_shot_shirt_design_image_prompts.ipynb)
* [Notebook Showcase: Keyword Extractor](https://github.com/FloTeu/llm-prompt-engineering-generator/blob/main/notebooks/keyword_extractor.ipynb)
* [App: Image Gen AI Prompt Generator](https://image-gen-ai-app.streamlit.app/)


## License
`llm-prompting-gen` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
