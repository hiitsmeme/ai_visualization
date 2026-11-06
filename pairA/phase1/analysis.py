"""Step 2: metrics computed from the saved CLIP outputs (results/scores.parquet).

Everything here works on model outputs only (logits), so it is valid for the black-box
Phase I. No model is loaded.

Measure: score vs. person = an occupation's logit minus the image's logit for "a photo of
a person". It says how much more CLIP sees "a photo of a nurse" in a face than just "a photo
of a person". A shift shared by all occupations (e.g. every occupation scoring higher for
men) stays visible, which centering on the image's mean occupation logit would cancel out.

Group gap: the mean score vs. person of one group of faces minus that of another (e.g. women
minus men), per occupation. All comparisons go through `mean_difference`.

Typical use (as in phase1_views.ipynb):
    df = load_scores()                         # one row per face
    values = scores_vs_person(df)              # faces x occupations
    gender_gap(df, values)                     # one row per occupation
    group_vs_rest(df, values, "race")          # one row per (ethnicity, occupation)
"""
import numpy as np
import pandas as pd

import config

Z_95 = 1.96  # standard normal quantile for a two-sided 95% confidence interval


# ---------------------------------------------------------------- loading
def load_scores(path=config.SCORES_PATH, age_groups=config.ANALYSIS_AGE_GROUPS):
    """Load the CLIP outputs, keeping only images in `age_groups` (default: people aged 20 and over).

    gender, race and age become categoricals in the order of config, so groupings, loops over
    `.cat.categories` and tables always list the groups in the same order.
    """
    df = pd.read_parquet(path)
    df = df[df["age"].isin(age_groups)].reset_index(drop=True)
    df["gender"] = pd.Categorical(df["gender"], categories=config.GENDERS)
    df["race"] = pd.Categorical(df["race"], categories=config.RACES)
    df["age"] = pd.Categorical(df["age"], categories=age_groups, ordered=True)
    return df


def prompt_family(df, family):
    """Logit columns of one prompt family ('occ', 'gender', 'race', 'age'), prefix removed.

    E.g. prompt_family(df, "gender") has the columns "Male" and "Female".
    """
    cols = [c for c in df.columns if c.startswith(family + ":")]
    return df[cols].rename(columns=lambda c: c.split(":", 1)[1])


def scores_vs_person(df):
    """Score vs. person: occupation logits minus the image's 'a photo of a person' logit (images x occupations)."""
    return prompt_family(df, config.OCC_FAMILY).sub(df[config.PERSON_COL], axis=0)


def top1(logits):
    """Column name of the highest logit per image, i.e. CLIP's zero-shot choice among the columns."""
    return logits.idxmax(axis=1)


# ---------------------------------------------------------------- sanity check
def _sanity_truth(df, family):
    """FairFace label of every image, in the label names of config.SANITY_PROMPTS[family].

    Gender and race prompts use the FairFace labels directly; the age prompts use coarse age
    words, so the age groups are mapped through config.AGE_TO_COARSE.
    """
    if family == "age":
        return df["age"].astype(str).map(config.AGE_TO_COARSE)
    return df[family].astype(str)


def sanity_confusion(df, family):
    """How often CLIP's zero-shot choice matches the FairFace label, for one family ('gender', 'race', 'age').

    CLIP "chooses" the sanity prompt of the family with the highest logit for each image.
    Returns (table, accuracy):
        table     rows = FairFace label, columns = CLIP's choice, values = share of the row
                  (each row sums to 1)
        accuracy  share of all images where CLIP's choice equals the FairFace label
    """
    labels = list(config.SANITY_PROMPTS[family])
    # Only the configured prompts compete: scores.parquet may hold older ones (e.g. 'baby', 'child').
    pred = top1(prompt_family(df, family)[labels])
    truth = _sanity_truth(df, family)
    counts = pd.crosstab(
        pd.Categorical(truth, labels), pd.Categorical(pred, labels),
        rownames=["FairFace label"], colnames=["CLIP zero-shot"], dropna=False,
    )
    # Drop labels without faces (e.g. 'teenager' once the under-20s are removed).
    counts = counts[counts.sum(axis=1) > 0]
    table = counts.div(counts.sum(axis=1), axis=0)
    accuracy = float((pred.to_numpy() == truth.to_numpy()).mean())
    return table, accuracy


# ---------------------------------------------------------------- view 2: score-level gaps
def mean_difference(values, in_a, in_b):
    """Per column of `values`: mean in group a minus mean in group b, with a 95% CI.

    values: DataFrame of images x columns (usually occupations, from scores_vs_person).
    in_a, in_b: boolean masks over the images (rows of `values`) that select the two groups.

    Returns one row per column of `values`, with the columns
        occupation        the column name
        diff              mean(a) - mean(b)
        ci_low, ci_high   95% confidence interval of diff
        n_a, n_b          number of images in each group

    The CI uses the normal approximation with Welch's standard error, which is
    accurate here because every group has hundreds to thousands of images.
    """
    in_a, in_b = np.asarray(in_a), np.asarray(in_b)  # plain arrays: select rows by position
    a, b = values[in_a], values[in_b]
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))  # Welch: groups may differ in size and variance
    return pd.DataFrame({
        "occupation": values.columns,
        "diff": diff.to_numpy(),
        "ci_low": (diff - Z_95 * se).to_numpy(),
        "ci_high": (diff + Z_95 * se).to_numpy(),
        "n_a": len(a),
        "n_b": len(b),
    })


def gender_gap(df, values):
    """Female minus male per occupation. Positive = CLIP leans to this occupation more for women.

    Returns the table of mean_difference (one row per occupation).
    """
    return mean_difference(values, df["gender"] == "Female", df["gender"] == "Male")


def group_vs_rest(df, values, col):
    """For each group of `col` ('race' or 'age'): mean in the group minus mean in all other groups.

    Returns the mean_difference tables of all groups stacked, with an extra 'group' column
    (one row per group and occupation).
    """
    parts = []
    for g in df[col].cat.categories:
        part = mean_difference(values, df[col] == g, df[col] != g)
        part["group"] = g
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def person_baseline(df, col="race"):
    """Distribution of the 'a photo of a person' logit per group, and each group's mean minus all other groups.

    Returns one row per group: group, n, mean, gap_vs_rest and the quantiles q05, q25, q50, q75, q95.
    """
    person = df[config.PERSON_COL]
    rows = []
    for g in df[col].cat.categories:
        sel = (df[col] == g).to_numpy()
        x = person[sel]
        rows.append({"group": g, "n": int(sel.sum()), "mean": x.mean(),
                     "gap_vs_rest": x.mean() - person[~sel].mean(),
                     **{f"q{q:02d}": x.quantile(q / 100) for q in (5, 25, 50, 75, 95)}})
    return pd.DataFrame(rows)


def _comparisons(df):
    """All comparisons of views 2a–2c as (name, in_a, in_b): women vs. men, each ethnicity
    vs. the rest and each age group vs. the rest."""
    comparisons = [("Female − Male", df["gender"] == "Female", df["gender"] == "Male")]
    for col, label in (("race", "{} − rest"), ("age", "age {} − rest")):
        comparisons += [(label.format(g), df[col] == g, df[col] != g) for g in df[col].cat.categories]
    return comparisons


def shared_shift(df, values):
    """Per comparison: the part of the gaps that all occupations share, and where it could come from.

    mean_gap: average of the comparison's gaps over all occupations.
    person_gap: the same comparison for the 'a photo of a person' logit itself. If the person
    prompt scores lower for a group, every occupation's score vs. person rises by that amount.
    occupation_gap = mean_gap + person_gap: the shift of the raw occupation logits
    (score vs. person = occupation logit - person logit, so occupation logit = score + person logit).

    Returns one row per comparison (index 'comparison', e.g. "Female − Male", "Black − rest").
    """
    person = df[[config.PERSON_COL]]  # one-column DataFrame, the input mean_difference expects
    rows = []
    for name, in_a, in_b in _comparisons(df):
        mean_gap = mean_difference(values, in_a, in_b)["diff"].mean()
        person_gap = mean_difference(person, in_a, in_b)["diff"].iloc[0]
        rows.append({"comparison": name, "mean_gap": mean_gap, "person_gap": person_gap,
                     "occupation_gap": mean_gap + person_gap})
    return pd.DataFrame(rows).set_index("comparison")
