from mid_prompt_gen.backend.midjourney import prompt_examples

def get_shirt_design_prompt_examples():
    return [prompt_examples.__dict__[item] for item in dir(prompt_examples) if item.startswith("shirt")]
