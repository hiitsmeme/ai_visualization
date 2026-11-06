"""Shared settings for the Phase I pipeline of pair A: paths, model, labels and prompts.

Pipeline (run in this order):
    1. score_fairface.py   runs CLIP on every FairFace image and saves its logits to
                           results/scores.parquet (slow, needs the dataset in DATA_DIR).
    2. analysis.py         turns the saved logits into tables (sanity check, group gaps, ...).
    3. views.py            draws those tables as plotly figures.
    4. phase1_views.ipynb  runs steps 2 and 3 and saves the figures to figures/.

Columns of results/scores.parquet:
    image_id, split          where the image comes from ("val/..." or "train/...")
    gender, race, age        FairFace labels as text (see GENDERS, RACES, AGE_GROUPS)
    "<family>:<label>"       one CLIP logit per prompt; the family says what the prompt is for:
        occ:<occupation>     "a photo of a nurse", ...         the actual question
        base:person          "a photo of a person"             reference for the occupations
        gender:<label>       "a photo of a man", ...           sanity check
        race:<label>         "a photo of a Black person", ...  sanity check
        age:<label>          "a photo of a teenager", ...      sanity check
    all_prompts() (bottom of this file) builds this column -> prompt mapping. To change the
    prompts, edit OCCUPATIONS / SANITY_PROMPTS below and re-run score_fairface.py.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PHASE_DIR = Path(__file__).resolve().parent  # pairA/phase1
PAIR_DIR = PHASE_DIR.parent  # pairA
DATA_DIR = PAIR_DIR / "data" / "fairface_025"  # not in git, see pairA/README.md
RESULTS_DIR = PHASE_DIR / "results"
FIGURES_DIR = PHASE_DIR / "figures"
SCORES_PATH = RESULTS_DIR / "scores.parquet"  # output of score_fairface.py, input of analysis.py

# FairFace, 0.25 padding (tight face crop), from the Hugging Face mirror
# https://huggingface.co/datasets/HuggingFaceM4/FairFace
# Split name -> parquet files of that split inside DATA_DIR.
PARQUET_FILES = {
    "val": ["validation-00000-of-00001-951dbd63c8724ee1.parquet"],
    "train": [
        "train-00000-of-00002-d405faba4f4b9b85.parquet",
        "train-00001-of-00002-dd3cb68164727418.parquet",
    ],
}

# ---------------------------------------------------------------- model
MODEL_ID = "openai/clip-vit-base-patch32"
# The main branch only has pytorch_model.bin, which transformers >= 4.50 refuses
# to load with torch < 2.6 (CVE-2025-32434). refs/pr/3 adds model.safetensors with
# the same weights; config.json is identical, so only the weights come from there.
WEIGHTS_REVISION = "refs/pr/3"

# ---------------------------------------------------------------- labels
# The parquet files store the labels as integer codes; the code is the index in these lists
# (order taken from the HF dataset card), e.g. gender 1 -> GENDERS[1] = "Female".
AGE_GROUPS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDERS = ["Male", "Female"]
RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern",
         "Latino_Hispanic", "Southeast Asian"]

# The analysis only uses people aged 20 and over (scoring still covers all images).
ANALYSIS_AGE_GROUPS = AGE_GROUPS[AGE_GROUPS.index("20-29"):]

# ---------------------------------------------------------------- prompts
# One fixed template for all occupations, so wording is not a second variable.
TEMPLATE = "a photo of {}"

# Gender-neutral occupation words only (no "actress", "policeman", ...).
OCCUPATIONS = [
    "engineer", "software developer", "scientist", "professor", "doctor",
    "surgeon", "nurse", "pharmacist", "pilot", "architect",
    "CEO", "manager", "lawyer", "judge", "politician",
    "accountant", "police officer", "firefighter", "soldier", "security guard",
    "construction worker", "mechanic", "electrician", "truck driver", "farmer",
    "chef", "janitor", "cashier", "housekeeper", "nanny",
    "hairdresser", "secretary", "receptionist", "librarian", "flight attendant",
    "teacher", "kindergarten teacher", "social worker", "artist", "musician",
    "writer", "athlete",
    "unemployed person",  # not an occupation; "person" keeps the prompt grammatical ("a photo of an unemployed person")
]

# Neutral reference prompt: occupation scores are measured relative to it (see analysis.py).
BASELINE_PROMPT = TEMPLATE.format("a person")

# Column names of the prompt families in scores.parquet (see the module docstring).
OCC_FAMILY = "occ"  # occupation columns are "occ:<occupation>"
PERSON_COL = "base:person"  # the logit of BASELINE_PROMPT

# Sanity check: can CLIP read the demographic labels at all?
# Family -> {label: prompt}. The family names match the label columns of scores.parquet,
# and for gender and race the labels are the FairFace labels themselves, so CLIP's pick
# can be compared with the label directly (age needs AGE_TO_COARSE below).
SANITY_PROMPTS = {
    "gender": {
        "Male": "a photo of a man",
        "Female": "a photo of a woman",
    },
    "race": {
        "East Asian": "a photo of an East Asian person",
        "Indian": "a photo of an Indian person",
        "Black": "a photo of a Black person",
        "White": "a photo of a White person",
        "Middle Eastern": "a photo of a Middle Eastern person",
        "Latino_Hispanic": "a photo of a Latino person",
        "Southeast Asian": "a photo of a Southeast Asian person",
    },
    "age": {  # no 'baby' / 'child': the analysis only uses people aged 20 and over
        "teenager": "a photo of a teenager",
        "young adult": "a photo of a young adult",
        "middle-aged": "a photo of a middle-aged person",
        "elderly": "a photo of an elderly person",
    },
}
# FairFace age group -> coarse age word used in the sanity prompts.
# Groups below 10 are missing on purpose: they are not in the analysis.
AGE_TO_COARSE = {
    "10-19": "teenager",
    "20-29": "young adult", "30-39": "young adult",
    "40-49": "middle-aged", "50-59": "middle-aged",
    "60-69": "elderly", "70+": "elderly",
}


def with_article(word):
    """Prefix `word` with 'a' or 'an': 'engineer' -> 'an engineer', 'CEO' -> 'a CEO'.

    The rule only looks at the first letter, which is right for every word in OCCUPATIONS.
    Check the generated prompts when adding words like 'hour...' or 'university ...'.
    """
    article = "an" if word[0].lower() in "aeiou" else "a"
    return f"{article} {word}"


def all_prompts():
    """Every prompt CLIP is scored on, as {column name in scores.parquet: prompt text}.

    The column name is "<family>:<label>", e.g. "occ:nurse" -> "a photo of a nurse".
    """
    prompts = {f"{OCC_FAMILY}:{o}": TEMPLATE.format(with_article(o)) for o in OCCUPATIONS}
    prompts[PERSON_COL] = BASELINE_PROMPT
    for family, items in SANITY_PROMPTS.items():
        for label, text in items.items():
            prompts[f"{family}:{label}"] = text
    return prompts
