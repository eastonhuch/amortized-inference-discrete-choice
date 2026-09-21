"""
Step 1: prepare the choice data.

Reads the panel, keeps households with MIN_PURCHASES..MAX_PURCHASES purchases,
builds the per-choice-set-size data dictionary, and saves it with its metadata
to <results root>/prepared/. The random draws are not stored; later steps
regenerate them from the seed and check them against the fingerprint saved
here.

    python step1_prepare_data.py            # skips if prepared data exist
    python step1_prepare_data.py --force    # rebuild

Note: prepared/data.pt is a processed copy of the panel data. Keep the results
root in the same protected location as the raw data and out of version control.
"""
import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import pipeline_common as pc
from analysis import PARAM_VERSION


def load_panel():
    """Read the panel and keep households in the purchase-count range."""
    from nielsen_shared import BASE_DIR

    df = pd.read_csv(os.path.join(BASE_DIR, pc.PANEL_FILE)).sort_values(
        by=["household_code", "purchase_date"]
    )
    household_value_counts = df["household_code"].value_counts()
    enough = household_value_counts >= pc.MIN_PURCHASES
    not_too_many = household_value_counts <= pc.MAX_PURCHASES
    print(f"Proportion of households with {pc.MIN_PURCHASES}+ purchases: {enough.mean()}")
    print(f"Proportion of households with <={pc.MAX_PURCHASES} purchases: {not_too_many.mean()}")

    households_in_range = household_value_counts.index[enough & not_too_many]
    if pc.HOUSEHOLD_SAMPLE_FRAC is not None:
        households_in_range = households_in_range.to_series().sample(
            frac=pc.HOUSEHOLD_SAMPLE_FRAC, random_state=pc.HOUSEHOLD_SAMPLE_SEED
        ).index
        print(f"Using a random {pc.HOUSEHOLD_SAMPLE_FRAC:.0%} sample of households")
    df = df[df["household_code"].isin(households_in_range)].copy()

    household_codes = np.sort(households_in_range)
    household_code_lookup = dict(zip(household_codes.tolist(), range(household_codes.size)))
    df["household_id"] = df["household_code"].map(household_code_lookup).astype(int)
    return df, household_value_counts


def plot_purchase_counts(household_value_counts, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    counts, bins, _ = ax[0].hist(
        household_value_counts, bins=np.arange(0, np.max(household_value_counts))
    )
    ax[0].set_xlabel("# Purchases")
    ax[0].set_ylabel("# Households")
    props = counts / counts.sum()
    df_counts = pd.DataFrame({
        "purchases": bins[:-1],
        "cumulative_proportion": props.cumsum(),
        "proportion": props,
        "cumulative_count": counts.cumsum(),
        "count": counts,
    })
    ax[1].plot(df_counts["purchases"], df_counts["cumulative_proportion"])
    ax[1].set_xlabel("# Purchases")
    ax[1].set_ylabel("Cumulative Proportion")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(df_counts.head(11).to_string(index=False))


def build_data_dict(df):
    """Brand-form columns, choice sets and the per-choice-set-size tensors (CPU)."""
    from nielsen_shared import get_brand_form

    df_brand_form = get_brand_form()
    brand_forms = df_brand_form["brand_form"].tolist()
    brand_form_lookup = dict(zip(brand_forms, range(len(brand_forms))))
    brand_form_id = df["brand_form"].map(brand_form_lookup)
    if brand_form_id.isna().any():
        unknown = sorted(df.loc[brand_form_id.isna(), "brand_form"].astype(str).unique())
        raise ValueError(f"Brand-forms missing from get_brand_form(): {unknown}")
    df["brand_form_id"] = brand_form_id.astype(int)
    brand_lookup = dict(df_brand_form[["brand_form", "brand"]].values)
    form_lookup = dict(df_brand_form[["brand_form", "form"]].values)
    form_lookup["OTHER"] = "OTHER"
    df["brand: OTHER"] = 0.
    df["form: OTHER"] = 0.

    loyalty_columns = [f"loyalty: {bf}" for bf in brand_forms]
    brand_loyalty_columns = [f"brand: {brand_lookup[bf]}" for bf in brand_forms]
    form_loyalty_columns = [f"form: {form_lookup[bf]}" for bf in brand_forms]
    coupon_columns = [f"coupon: {bf}" for bf in brand_forms]
    price_columns = [f"price: {bf}" for bf in brand_forms]
    promotion_columns = [f"promotion: {bf}" for bf in brand_forms]

    # Price and promotion are missing when an alternative is unavailable
    bf_is_available = ~df[price_columns].isna()
    df["choice_set_size"] = bf_is_available.sum(axis=1)

    choice_set_sizes = np.sort(df["choice_set_size"].unique()).tolist()
    data_dict = {}
    X_dim = None
    for c in choice_set_sizes:
        df_c = df[df["choice_set_size"] == c]
        available_c = (~df_c[price_columns].isna()).to_numpy()
        # Column indices of the available alternatives, row by row (sorted)
        alternative_ids_c_np = np.nonzero(available_c)[1].reshape(-1, c)
        X_full_c = np.dstack([
            np.ones_like(df_c[loyalty_columns]),
            df_c[loyalty_columns],
            df_c[brand_loyalty_columns],
            df_c[form_loyalty_columns],
            df_c[coupon_columns],
            df_c[price_columns],
            df_c[promotion_columns],
        ])
        X_np = X_full_c[np.arange(alternative_ids_c_np.shape[0])[:, None], alternative_ids_c_np]
        X_dim = X_np.shape[2]
        if np.isnan(X_np).any():
            raise ValueError(
                f"Missing covariate values for available alternatives (choice-set size {c})."
            )

        alternative_ids_c = torch.tensor(alternative_ids_c_np, dtype=torch.int64)
        choice_ids_c = torch.tensor(df_c["brand_form_id"].values, dtype=torch.int64)
        chosen_available = alternative_ids_c == choice_ids_c.unsqueeze(1)
        n_unavailable = int((~chosen_available.any(dim=1)).sum())
        if n_unavailable > 0:
            print(
                f"WARNING: {n_unavailable} purchases with choice-set size {c} chose a "
                "brand-form that is not in the choice set; their choice index is set "
                "to 0."
            )
        data_dict[c] = {
            "alternative_ids": alternative_ids_c,
            "X": torch.tensor(X_np, dtype=torch.float32),
            "household_id": torch.tensor(df_c["household_id"].values, dtype=torch.int64),
            "choice_id": choice_ids_c,
            "choice_idx": torch.max(chosen_available, dim=1).indices,
        }

    columns = {
        "loyalty": loyalty_columns, "coupon": coupon_columns,
        "price": price_columns, "promotion": promotion_columns,
    }
    return data_dict, brand_forms, choice_set_sizes, X_dim, columns


def plot_covariates(df, columns, path):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    for i, key in enumerate(["loyalty", "coupon", "price", "promotion"]):
        ax[i].hist(df[columns[key]].values.flatten(), bins=np.linspace(0., 1., 101), density=True)
        ax[i].set_yscale("log")
        ax[i].set_xlabel(key)
    ax[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--force", action="store_true", help="rebuild the prepared data")
    args = parser.parse_args(argv)

    pc.start_logging("step1_prepare_data")
    try:
        if os.path.exists(pc.prepared_metadata_path()) and not args.force:
            print(f"Prepared data already exist in {pc.PREPARED_DIR}; nothing to do "
                  "(use --force to rebuild).")
            return

        pc.banner("STEP 1: PREPARE DATA")
        figures_dir = os.path.join(pc.PREPARED_DIR, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        df, household_value_counts = load_panel()
        plot_purchase_counts(household_value_counts, os.path.join(figures_dir, "purchase_counts.pdf"))
        data_dict, brand_forms, choice_set_sizes, X_dim, columns = build_data_dict(df)
        plot_covariates(df, columns, os.path.join(figures_dir, "covariate_distributions.pdf"))

        if X_dim != len(pc.NUMERIC_NAMES) + 1:
            raise ValueError(f"X_dim={X_dim} does not match NUMERIC_NAMES ({len(pc.NUMERIC_NAMES)} + intercept).")

        n_households = int(df["household_id"].max()) + 1
        max_k = (len(brand_forms) - 1) + len(pc.RANDOM_FEATURE_INDICES)
        print(f"Generating random draws: {n_households} households x {pc.N_DRAWS} draws x {max_k} dimensions")
        draws = pc.make_std_normal_draws(n_households, pc.N_DRAWS, max_k, pc.RANDOM_SEED)

        data_tensors = []
        for c in choice_set_sizes:
            data_tensors += [data_dict[c][key] for key in sorted(data_dict[c])]
        fingerprint_inputs = {
            "data_fingerprint": pc.fingerprint_tensors(data_tensors),
            "draws_fingerprint": pc.fingerprint_tensors([draws]),
            "n_draws": pc.N_DRAWS,
            "random_seed": pc.RANDOM_SEED,
            "random_feature_indices": pc.RANDOM_FEATURE_INDICES,
            "param_version": PARAM_VERSION,
        }
        metadata = {
            "n_observations": int(df.shape[0]),
            "n_households": n_households,
            "n_brand_forms": len(brand_forms),
            "X_dim": int(X_dim),
            "brand_forms": brand_forms,
            "intercept_names": brand_forms[1:],
            "numeric_names": pc.NUMERIC_NAMES,
            "random_feature_indices": pc.RANDOM_FEATURE_INDICES,
            "choice_set_sizes": [int(c) for c in choice_set_sizes],
            "n_draws": pc.N_DRAWS,
            "random_seed": pc.RANDOM_SEED,
            "max_k": max_k,
            "data_fingerprint": fingerprint_inputs["data_fingerprint"],
            "draws_fingerprint": fingerprint_inputs["draws_fingerprint"],
            "fingerprint_inputs": fingerprint_inputs,
            "run_fingerprint": pc.fingerprint_values(fingerprint_inputs),
            "household_filter": {
                "min_purchases": pc.MIN_PURCHASES,
                "max_purchases": pc.MAX_PURCHASES,
                "sample_frac": pc.HOUSEHOLD_SAMPLE_FRAC,
                "sample_seed": pc.HOUSEHOLD_SAMPLE_SEED,
            },
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        pc.save_prepared_data(data_dict, metadata)
        print(f"\nSaved prepared data to {pc.PREPARED_DIR}")
        print(f"  {metadata['n_observations']} purchases, {n_households} households, "
              f"{len(brand_forms)} brand-forms, choice-set sizes {metadata['choice_set_sizes']}")
        print(f"  Run fingerprint: {metadata['run_fingerprint']}")
    finally:
        pc.stop_logging()


if __name__ == "__main__":
    main()
