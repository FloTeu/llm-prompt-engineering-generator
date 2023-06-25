# LLM Few Shot Gen

[![PyPI - Version](https://img.shields.io/pypi/v/llm-few-shot-gen.svg)](https://pypi.org/project/llm-few-shot-gen)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm-few-shot-gen.svg)](https://pypi.org/project/llm-few-shot-gen)

-----


## Installation

```console
pip install llm-few-shot-gen
```

## What is LLM Few Shot Generator?
Large language models (LLMs) are emerging as a transformative technology, enabling developers to build applications that they previously could not. However, fine-tune them is not always possible or too expansive. In-context learning like Few Shot learning tries to solve this problem. Based on a few number of examples within the prompt a specific output can be obtained. This llm-few-shot-gen library enables few shot learning in a convenient way. 

## How can I use it?
The core functionality is the `generators` module. The abstract class `BaseFewShotGenerator` defines all requirements for a LLM few shot generator. You can either use a base class and create a child for your own use case or pick one of the existing generator classes. 

### Required functions of `BaseFewShotGenerator` children

**Instruction**:
Instruction for LLM model

**Context**:
Context of the Few Shot learning use case

**Few Shot Examples**:
Example text to obtain a certain output

**IO Prompt**:
Output instruction, which might contain some human input as well. I/O=input/output


### Ready to use Generators
* [Midjourney Prompt Generator](https://github.com/FloTeu/llm-few-shot-generator/blob/main/src/llm_few_shot_gen/generators/midjourney.py)



### Showcases
* [Notebook Showcase](https://github.com/FloTeu/llm-few-shot-generator/blob/main/notebooks/few_shot_shirt_designs.ipynb)
* [Midjourney Prompt Generator](https://midjourney-prompt-generator.streamlit.app/)


## License

`llm-few-shot-gen` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
