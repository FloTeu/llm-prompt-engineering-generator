from llm_few_shot_gen.few_shot_examples import midjourney

def get_shirt_design_prompt_examples():
    return [midjourney.__dict__[item] for item in dir(midjourney) if item.startswith("shirt")]
