# Does a face tell CLIP your job?

## Task

We look for social bias in CLIP's zero-shot classification of faces. CLIP gets a face and prompts like "a photo of a nurse". A face says nothing about a job, so if the scores depend on gender, age or ethnicity, we count that as bias. Phase I treats CLIP as a black box (inputs and output scores only). Phase II looks inside, mainly at the shared image–text embedding space.

## Model

[CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) ([paper](https://arxiv.org/abs/2103.00020), [repo](https://github.com/openai/CLIP)) via Hugging Face Transformers, zero-shot, no fine-tuning.

## Dataset

[FairFace](https://github.com/joojs/fairface) ([paper](https://arxiv.org/abs/1908.04913)): faces balanced over 7 ethnicity groups, labeled with gender and 9 age groups. We use the 0.25 padding version from [`HuggingFaceM4/FairFace`](https://huggingface.co/datasets/HuggingFaceM4/FairFace) (train + validation, 97,698 images) and analyze the 73,659 faces aged 20+.
