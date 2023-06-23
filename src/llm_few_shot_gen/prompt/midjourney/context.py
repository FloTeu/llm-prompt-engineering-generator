"""
Collection of Midjourney information regarding Midjourney Tools, Prompt Engineering etc.
"""

# extracted from: https://docs.midjourney.com/docs/prompts
prompt_general_description = "A Prompt is a short text phrase that the Midjourney Bot interprets to produce an image. The Midjourney Bot breaks down the words and phrases in a prompt into smaller pieces, called tokens, that can be compared to its training data and then used to generate an image. A well-crafted prompt can help make unique and exciting images."
prompt_length = "Prompts can be very simple. Single words (or even an emoji!) will produce an image. Very short prompts will rely heavily on Midjourney’s default style, so a more descriptive prompt is better for a unique look. However, super-long prompts aren’t always better. Concentrate on the main concepts you want to create."
prompt_grammer = "The Midjourney Bot does not understand grammar, sentence structure, or words like humans. Word choice also matters. More specific synonyms work better in many circumstances. Instead of big, try gigantic, enormous, or immense. Remove words when possible. Fewer words mean each word has a more powerful influence. Use commas, brackets, and hyphens to help organize your thoughts, but know the Midjourney Bot will not reliably interpret them. The Midjourney Bot does not consider capitalization. Midjourney Model Version 4 is slightly better than other models at interpreting traditional sentence structure."
prompt_what_you_want = "It is better to describe what you want instead of what you don’t want. If you ask for a party with “no cake,” your image will probably include a cake. If you want to ensure an object is not in the final image, try advance prompting using the --no parameter."
prompt_details = """Anything left unsaid may suprise you. Be as specific or vague as you want, but anything you leave out will be randomized. Being vague is a great way to get variety, but you may not get the specific details you want.

Try to be clear about any context or details that are important to you. Think about:

Subject: person, animal, character, location, object, etc.
Medium: photo, painting, illustration, sculpture, doodle, tapestry, etc.
Environment: indoors, outdoors, on the moon, in Narnia, underwater, the Emerald City, etc.
Lighting: soft, ambient, overcast, neon, studio lights, etc
Color: vibrant, muted, bright, monochromatic, colorful, black and white, pastel, etc.
Mood: Sedate, calm, raucous, energetic, etc.
Composition: Portrait, headshot, closeup, birds-eye view, etc.
"""

# https://www.crunchbase.com/organization/midjourney
midjourney_company_information = "Midjourney is an artificial intelligence-powered artwork generator. It explores new thought mediums and expands the human species' imaginative powers. It is a small self-funded team focused on design, human infrastructure, and artificial intelligence."
# https://docs.midjourney.com/legacy/docs
midjourney_v5_general_description = "Midjourney routinely releases new model versions to improve efficiency, coherency, and quality. The latest model is the default, but other models can be used using the --version or --v parameter or by using the /settings command and selecting a model version. Different models excel at different types of images."
midjourney_v5_additional_description = "The Midjourney V5 model is the newest and most advanced model, released on March 15th, 2023. To use this model, add the --v 5 parameter to the end of a prompt, or use the /settings command and select 5️⃣ MJ Version 5. This model has very high Coherency, excels at interpreting natural language prompts, is higher resolution, and supports advanced features like repeating patterns with --tile"
midjourney_model_switch = """
How to Switch Models
Use the Version or Test Parameter
Add --v 4 --v 5 --v 5.1 --v 5.1 --style raw --v 5.2 --v 5.2 --style raw --niji 5 --niji 5 --style cute --niji 5 --style expressive --niji 5 --style original or --niji 5 --style scenic to the end of your prompt.
"""
