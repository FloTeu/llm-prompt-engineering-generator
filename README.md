# LLM Few Shot Gen

[![PyPI - Version](https://img.shields.io/pypi/v/llm-prompting-gen.svg)](https://pypi.org/project/llm-prompting-gen)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm-prompting-gen.svg)](https://pypi.org/project/llm-prompting-gen)

-----


> [!WARNING]  
> This library is not maintained anymore. Checkout the new project llm-prompting-gen.

## Installation

```console
pip install llm-prompting-gen
```

## What is LLM Few Shot Generator?
Large language models (LLMs) are emerging as a transformative technology, enabling developers to build applications that they previously could not. However, fine-tune them is not always possible or too expansive. In-context learning like Few Shot learning tries to solve this problem. Based on a few number of examples within the prompt a specific output can be obtained. This llm-prompting-gen library enables few shot learning in a convenient way. 

## How can I use it?
The core functionality is the `generators` module. The class `PromptEngineeringGenerator` defines all requirements for a LLM to generate text based on prompt engineering techniques. If you want your output to be parsed into an pydantic dataclass checkout the class `ParsablePromptEngineeringGenerator`. 

## How to customise the class for my own use case?
The class `PromptEngineeringGenerator` contains two core parts 1. LLM 2. prompt engineering dataclass. If you want to initialise the generator class for you custom use case, you need to define a prompt engineering JSON file matching the format of `llm_prompting_gen.models.prompt_engineering.PromptElements`.

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
* [Notebook Showcase: Midjourney Prompt](https://github.com/FloTeu/llm-prompt-engineering-generator/blob/main/notebooks/few_shot_shirt_designs.ipynb)
* [Notebook Showcase: Keyword Extractor](https://github.com/FloTeu/llm-prompt-engineering-generator/blob/main/notebooks/few_shot_keyword_extractor.ipynb)
* [App: Image Gen AI Prompt Generator](https://image-gen-ai-app.streamlit.app/)


## License

`llm-prompting-gen` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
