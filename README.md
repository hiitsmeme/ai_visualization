# Does a face tell CLIP your job?

Lab assignment for the AI Visualization course at JKU Linz (2026), team **AI Vizards**.

We investigate social biases in zero-shot image classification with CLIP. We show CLIP face images from the FairFace dataset together with text prompts that name occupations. A face carries no information about a person's occupation, so an unbiased model's scores should not depend on the gender, age or ethnicity of the face. We treat any systematic dependence as a bias.

## Team

| Pair | Members | Phase I | Phase II |
|---|---|---|---|
| A | Sonja Topf, Sebastian Kranzinger | without GenAI | with GenAI |
| B | Rareș Cazan, Diana-Alexandra Maxiniuc | with GenAI | without GenAI |

Student IDs are listed in [`team-info.json`](team-info.json), the workload split in [`workload.json`](workload.json).

## Task

The assignment has two phases that ask complementary questions:

- **Phase I: How does the model behave?** We treat CLIP as a black box and only look at its inputs (faces and prompts) and outputs (image–text scores). We search for occupations that CLIP associates with particular groups and test how robust these associations are.
- **Phase II: What did the model learn?** We open the black box and examine CLIP's internal representations, in particular its shared image–text embedding space, to understand how the associations found in Phase I are encoded.

In each phase, one pair works with the support of generative AI and the other pair works without it. The pairs switch roles after Phase I and compare their processes and findings at the end of each phase (AI4Vis).

## Model

- **CLIP ViT-B/32** by OpenAI, a pretrained image–text model, used zero-shot via Hugging Face Transformers without fine-tuning.
  - Model: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
  - Paper: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021 ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
  - Repository: [github.com/openai/CLIP](https://github.com/openai/CLIP)
- **Output used in Phase I:** `logits_per_image`, i.e. CLIP's logit scale (100 for this model) times the cosine similarity between an image and a prompt. The Phase I analyses of both pairs use only these outputs, no embeddings.
- Pair A loads the weights from the `refs/pr/3` revision of the model repository, which holds the same weights as `model.safetensors`. The main branch only has `pytorch_model.bin`, which `transformers` ≥ 4.50 refuses to load with `torch` < 2.6.

## Dataset

- **FairFace** (Kärkkäinen & Joo, *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age*, 2021; [arXiv:1908.04913](https://arxiv.org/abs/1908.04913), [github.com/joojs/fairface](https://github.com/joojs/fairface)).
- Face images balanced across 7 ethnicity groups (East Asian, Southeast Asian, Indian, Black, White, Middle Eastern, Latino/Hispanic), each labeled with gender (male, female) and one of 9 age groups (0–2 to 70+). The labels come from human annotators and describe perceived attributes, not self-reported ones.
- **Version:** the 0.25 padding variant (tight face crop) from the Hugging Face mirror [`HuggingFaceM4/FairFace`](https://huggingface.co/datasets/HuggingFaceM4/FairFace): 86,744 training and 10,954 validation images, 97,698 in total.
- **Subset analyzed:** training and validation images of people aged 20 and over, **73,659 faces** (39,390 labeled male, 34,269 labeled female). Both pairs use exactly this set.

## Phase I: How does the model behave?

### Pair A, without GenAI ([`pairA/phase1/`](pairA/phase1))

- **Prompts:** 43 gender-neutral occupations with one fixed template, `"a photo of a/an <occupation>"`, plus `"a photo of a person"` as a neutral reference.
- **Measure:** *score vs. person* = an occupation's logit minus the same image's logit for `"a photo of a person"`. A *group gap* is the mean score vs. person of one group minus that of another (women vs. men, each ethnicity vs. all others, each age group vs. all others), with a 95% confidence interval (normal approximation, Welch standard error).
- **Sanity checks:** whether CLIP can tell gender, ethnicity and age apart at all (its zero-shot pick vs. the FairFace label), and how well the reference prompt fits each ethnicity.
- **Views:** interactive Plotly figures in [`pairA/phase1/figures/`](pairA/phase1/figures). Open the HTML files in a browser and hover a dot to see its occupation, gap, rank and group sizes.
  - `0_sanity.html`: CLIP's picks vs. the FairFace labels
  - `2a_gender_gap.html`: occupations associated with women's or men's faces
  - `2b_ethnicity_vs_rest.html`: each ethnicity vs. all others
  - `2c_age_vs_rest.html`: each age group vs. all others
  - `2d_person_baseline.html`: how well `"a photo of a person"` fits each ethnicity

### Pair B, with GenAI ([`PairB/`](PairB))

- **Prompts:** 8 occupations (doctor, nurse, engineer, scientist, professor, teacher, artist, athlete) with the template `"a portrait photo of a/an <occupation>"`.
- **Measure:** the *top pick*, the occupation with the highest logit for a face, and the share of faces in each group that receive each occupation.
- **Analyses** (all in [`CLIP_Bias_Analysis.ipynb`](PairB/CLIP_Bias_Analysis.ipynb)):
  - selection rates by gender and ethnicity, and within gender and age groups
  - the engineer gender gap with a bootstrap interval and a comparison within age × ethnicity subgroups
  - a glasses experiment: a hand-labeled sample of faces with and without visible glasses ([`glasses_labels.csv`](PairB/glasses_labels.csv)) and a regression that controls for gender, age and ethnicity
  - prompt wording: three templates for the same occupations
  - the candidate list: how results change when nurse or athlete is removed
  - a PCA of the eight occupation scores

### GenAI documentation

- [`ai-interactions/`](ai-interactions): Pair B's record of the interactions with GenAI that shaped the analysis.
- [`ai_reflection_midpoint.md`](ai_reflection_midpoint.md): Pair B's reflection on working with GenAI in Phase I.

## Midpoint presentation

The Phase I results of both pairs and the AI4Vis comparison are summarized in the slides [`MidpointPresentation.pdf`](MidpointPresentation.pdf) and the recorded talk [`MidpointPresentation.mp4`](MidpointPresentation.mp4).

## Repository structure

```
.
├── README.md
├── MidpointPresentation.pdf      Phase I midpoint slides
├── MidpointPresentation.mp4      recording of the midpoint presentation
├── team-info.json
├── workload.json
├── ai_reflection_midpoint.md
├── ai-interactions/
│   └── Interactions.txt
├── pairA/
│   └── phase1/
│       ├── config.py              paths, model, labels and prompts
│       ├── score_fairface.py      step 1: runs CLIP on all images, saves the logits
│       ├── analysis.py            step 2: tables from the saved logits
│       ├── views.py               step 3: Plotly figures from the tables
│       ├── phase1_views.ipynb     runs steps 2 and 3, saves the figures
│       ├── results/scores.parquet saved CLIP logits
│       └── figures/               interactive HTML views
└── PairB/
    ├── CLIP_Bias_Analysis.ipynb   the whole Phase I analysis
    ├── glasses_labels.csv         hand labels for the glasses experiment
    ├── *.png, clip_results.csv    saved figures and scores
    └── *.npy, embedding_*.png     from an earlier embedding-based run, not used by the notebook
```

## Reproducing the results

### Pair A

Python 3.12 with `torch`, `transformers`, `pandas`, `pyarrow`, `pillow`, `numpy`, `plotly` and Jupyter.

1. Download the three parquet files from the [`0.25` folder](https://huggingface.co/datasets/HuggingFaceM4/FairFace/tree/main/0.25) of the Hugging Face mirror into `pairA/data/fairface_025/` (not tracked by git).
2. In `pairA/phase1/`, run `py -3.12 score_fairface.py` to score all 97,698 images and write `results/scores.parquet`. `--limit 512` does a quick test run.
3. Run `phase1_views.ipynb` to compute the tables and save the figures to `figures/`.

Step 2 can be skipped, since `results/scores.parquet` is in the repository.

### Pair B

Python 3.10 with `torch`, `transformers`, `datasets`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels` and `tqdm`. The notebook downloads FairFace through `datasets` itself.

Run `PairB/CLIP_Bias_Analysis.ipynb` from the top, with `glasses_labels.csv` in the working directory. Scoring all faces against 24 prompts is much faster on a GPU.
