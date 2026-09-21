"""
Step 2: exact (non-emulator) models.

For each family (mixed probit, then mixed logit):
  1. factor models with n_factors_rand in N_FACTORS_RAND_LIST, each warm-started
     from the previous one;
  2. after both sequences, the dense model (n_factors_rand = k, no
     uniquenesses), warm-started from the family's best factor model by BIC.

Each model is saved to <results root>/mnl_models/ as soon as it is fit, and its
coefficient table is created right after. Rerunning skips everything that is
already on disk.

    python step2_fit_exact_models.py
    python step2_fit_exact_models.py --families probit     # one family only
    python step2_fit_exact_models.py --no-coef-tables      # fits only (rerun later to add tables)
"""
import argparse
import gc
import os
import time

import pandas as pd

import pipeline_common as pc
from analysis import (
    convert_to_dense_params,
    empty_cache,
    fit_single_model,
    get_param_names_from_kwargs,
    get_params_from_summary,
    result_dict_to_summary,
    save_model_summary,
)


def _print_fit(summary: dict) -> None:
    m = pc.summary_metrics(summary)
    print(f"  Log-likelihood: {m['loglike']:.2f} | BIC: {m['bic']:.2f} | "
          f"n_params: {m['n_params']} | Converged: {m['converged']} | "
          f"Time: {m['elapsed_time']:.1f}s")


def _coef_timing_path() -> str:
    return os.path.join(pc.EXACT_RESULTS_DIR, "coef_table_timing_summary.json")


def fit_factor_models(ctx: pc.AnalysisContext, family: str, coef_tables: bool) -> list:
    """Fit (or load) the factor-model sequence of one family. Returns metric rows."""
    label = pc.FAMILIES[family]["label"]
    prefix = pc.FAMILIES[family]["prefix"]
    pc.banner(f"FITTING {label.upper()} MODELS")
    model_kwargs = ctx.model_kwargs(family, include_uniquenesses_rand=True)

    rows = []
    prev_params, prev_n_factors, prev_path = None, -1, None
    for idx, n_factors in enumerate(pc.N_FACTORS_RAND_LIST):
        path = pc.factor_model_path(family, n_factors)
        progress = f"[{idx + 1}/{len(pc.N_FACTORS_RAND_LIST)}]"
        summary = pc.load_summary_if_exists(path)

        if summary is None:
            source = os.path.basename(prev_path) if prev_path else "zeros"
            print(f"\n{progress} Fitting {label.lower()} model with {n_factors} factors "
                  f"(warm start: {source})")
            start = time.time()
            fit_result = fit_single_model(
                n_factors=n_factors,
                batch_data_dicts=ctx.batches(pc.N_BATCHES_FACTOR_FIT),
                start_param=prev_params,
                start_param_n_factors_rand=prev_n_factors,
                model_kwargs=model_kwargs,
                n_households=ctx.n_households,
                device=ctx.device,
                n_factors_time=-1,
                start_param_n_factors_time=-1,
                n_random_restarts=pc.N_RANDOM_RESTARTS,
                restart_noise_std=pc.RESTART_NOISE_STD,
            )
            elapsed = time.time() - start
            summary = result_dict_to_summary(
                fit_result,
                model_name=f"{prefix}_mixed_factors_{n_factors}",
                param_names=get_param_names_from_kwargs(
                    model_kwargs, n_factors, -1, ctx.intercept_names, ctx.numeric_names
                ),
            )
            summary["elapsed_time"] = elapsed
            summary["warm_start"] = {"source": source}
            save_model_summary(summary, path)
            del fit_result
            gc.collect()
            empty_cache(ctx.device)
        else:
            print(f"\n{progress} Loaded {path}")
        _print_fit(summary)

        if coef_tables:
            pc.ensure_coef_table(
                ctx, path, pc.exact_coef_table_path(path), model_kwargs,
                _coef_timing_path(), {"model_family": family, "n_factors": n_factors},
            )

        rows.append({"n_factors": n_factors, **pc.summary_metrics(summary)})
        pc.plot_factor_comparison(
            rows, family, os.path.join(pc.EXACT_RESULTS_DIR, f"{prefix}_factor_comparison.pdf")
        )
        prev_params = get_params_from_summary(summary, device=ctx.device)
        prev_n_factors, prev_path = n_factors, path

    comparison = {
        "n_factors_list": [row["n_factors"] for row in rows],
        "log_likelihoods": [row["loglike"] for row in rows],
        "aics": [row["aic"] for row in rows],
        "bics": [row["bic"] for row in rows],
        "caics": [row["caic"] for row in rows],
        "times": [row["elapsed_time"] for row in rows],
        "total_time": float(sum(row["elapsed_time"] for row in rows)),
    }
    save_model_summary(
        comparison, os.path.join(pc.EXACT_RESULTS_DIR, f"{family}_comparison_summary.json")
    )
    return rows


def best_by_bic(rows: list) -> tuple:
    best = min(rows, key=lambda row: row["bic"])
    return best["n_factors"], best["bic"]


def report_bic_selection(rows_by_family: dict) -> dict:
    pc.banner("OPTIMAL NUMBER OF FACTORS (BIC)")
    best = {}
    for family, rows in rows_by_family.items():
        label = pc.FAMILIES[family]["label"]
        series = pd.Series({row["n_factors"]: row["bic"] for row in rows}, name="BIC")
        series.index.name = "n_factors"
        print(f"\n{label} BIC by number of factors:")
        print(series.to_string())
        best[family] = best_by_bic(rows)
        print(f"Optimal {label} model: {best[family][0]} factors (BIC = {best[family][1]:.2f})")

    if len(rows_by_family) == 2:
        bic_df = pd.DataFrame({
            "Logit BIC": {row["n_factors"]: row["bic"] for row in rows_by_family["logit"]},
            "Probit BIC": {row["n_factors"]: row["bic"] for row in rows_by_family["probit"]},
        })
        bic_df.index.name = "n_factors"
        bic_df["Logit - Probit"] = bic_df["Logit BIC"] - bic_df["Probit BIC"]
        print("\nBIC comparison (Logit vs Probit):")
        print(bic_df.to_string())

        overall = {"n_factors_list": [int(n) for n in bic_df.index.tolist()]}
        for family, rows in rows_by_family.items():
            overall[family] = {
                "log_likelihoods": [row["loglike"] for row in rows],
                "aics": [row["aic"] for row in rows],
                "bics": [row["bic"] for row in rows],
                "caics": [row["caic"] for row in rows],
                "times": [row["elapsed_time"] for row in rows],
                "total_time": float(sum(row["elapsed_time"] for row in rows)),
            }
        overall["total_time"] = overall["probit"]["total_time"] + overall["logit"]["total_time"]
        save_model_summary(
            overall, os.path.join(pc.EXACT_RESULTS_DIR, "overall_comparison_summary.json")
        )
    return best


def fit_dense_model(ctx: pc.AnalysisContext, family: str, best_n_factors: int,
                    best_bic: float, coef_tables: bool) -> dict:
    """Fit (or load) the dense model of one family, warm-started from its best factor model."""
    label = pc.FAMILIES[family]["label"]
    prefix = pc.FAMILIES[family]["prefix"]
    pc.banner(f"Dense {label} model (n_factors=k={ctx.k_rand}, no uniquenesses)", char="-")
    model_kwargs = ctx.model_kwargs(family, include_uniquenesses_rand=False)
    path = pc.dense_model_path(family)
    source_path = pc.factor_model_path(family, best_n_factors)
    summary = pc.load_summary_if_exists(path)

    if summary is None:
        source_summary = pc.require_summary(source_path, "Fit the factor models first.")
        source_params = get_params_from_summary(source_summary, device=ctx.device)
        start_params = convert_to_dense_params(
            source_params=source_params,
            source_n_factors_rand=best_n_factors,
            n_brand_forms=ctx.n_brand_forms,
            n_fixed_features=ctx.n_fixed_features,
            random_feature_indices=ctx.random_feature_indices,
            include_uniquenesses_rand=best_n_factors >= 0,
            device=ctx.device,
            anchored_rand=True,
        )
        print(f"Warm-starting from {os.path.basename(source_path)} "
              f"({len(source_params)} -> {len(start_params)} parameters)")
        start = time.time()
        fit_result = fit_single_model(
            n_factors=ctx.k_rand,
            batch_data_dicts=ctx.batches(pc.N_BATCHES_DENSE_FIT),
            start_param=start_params,
            start_param_n_factors_rand=ctx.k_rand,
            start_param_n_factors_time=-1,
            model_kwargs=model_kwargs,
            n_households=ctx.n_households,
            device=ctx.device,
            n_factors_time=-1,
            n_random_restarts=pc.N_RANDOM_RESTARTS,
            restart_noise_std=pc.RESTART_NOISE_STD,
        )
        elapsed = time.time() - start
        summary = result_dict_to_summary(
            fit_result,
            model_name=f"{prefix}_mixed_dense",
            param_names=get_param_names_from_kwargs(
                model_kwargs, ctx.k_rand, -1, ctx.intercept_names, ctx.numeric_names
            ),
        )
        summary["elapsed_time"] = elapsed
        summary["warm_start"] = {"source": os.path.basename(source_path)}
        save_model_summary(summary, path)
        del fit_result, start_params, source_params
        gc.collect()
        empty_cache(ctx.device)
    else:
        print(f"Loaded {path}")
        used = summary.get("warm_start", {}).get("source")
        if used is not None and used != os.path.basename(source_path):
            print(f"  NOTE: this dense model was warm-started from {used}, but the best "
                  f"factor model is now {os.path.basename(source_path)}. Delete {path} to refit.")
    _print_fit(summary)

    dense_bic = float(summary["bic"])
    print(f"  Best factor model (n_factors={best_n_factors}) BIC: {best_bic:.4f}; "
          f"dense - factor: {dense_bic - best_bic:.4f} -> "
          f"{'dense' if dense_bic < best_bic else 'factor'} model preferred by BIC")

    if coef_tables:
        pc.ensure_coef_table(
            ctx, path, pc.exact_coef_table_path(path), model_kwargs,
            _coef_timing_path(), {"model_family": family, "n_factors": "dense"},
        )
    return {**pc.summary_metrics(summary), "best_factor_n_factors": int(best_n_factors),
            "best_factor_bic": float(best_bic)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 2: exact mixed probit/logit models")
    parser.add_argument("--families", nargs="+", choices=["probit", "logit"],
                        default=["probit", "logit"])
    parser.add_argument("--no-coef-tables", action="store_true",
                        help="skip coefficient tables (a later rerun adds the missing ones)")
    parser.add_argument("--skip-dense", action="store_true", help="fit the factor models only")
    args = parser.parse_args(argv)

    pc.start_logging("step2_fit_exact_models")
    try:
        ctx = pc.AnalysisContext()
        pc.check_results_dir(pc.EXACT_RESULTS_DIR, ctx.metadata)
        coef_tables = not args.no_coef_tables
        step_start = time.time()

        rows_by_family = {}
        for family in [f for f in ["probit", "logit"] if f in args.families]:
            rows_by_family[family] = fit_factor_models(ctx, family, coef_tables)
            ctx.release_batches(pc.N_BATCHES_FACTOR_FIT)

        best = report_bic_selection(rows_by_family)
        if args.skip_dense:
            return

        pc.banner("FITTING DENSE COVARIANCE MODELS")
        dense = {}
        for family in rows_by_family:
            dense[family] = fit_dense_model(ctx, family, *best[family], coef_tables)

        pc.banner("DENSE MODELS SUMMARY")
        dense_df = pd.DataFrame({
            pc.FAMILIES[family]["label"] + " (Dense)": {
                "Log-likelihood": m["loglike"], "n_params": m["n_params"], "AIC": m["aic"],
                "BIC": m["bic"], "CAIC": m["caic"], "Converged": m["converged"],
                "Time (s)": m["elapsed_time"], "Best factor n_factors": m["best_factor_n_factors"],
                "Best factor BIC": m["best_factor_bic"],
                "Preferred": "Dense" if m["bic"] < m["best_factor_bic"] else "Factor",
            }
            for family, m in dense.items()
        }).T
        print(dense_df.to_string())
        if len(dense) == 2:
            save_model_summary(
                {family: {
                    "log_likelihood": m["loglike"], "n_params": m["n_params"], "aic": m["aic"],
                    "bic": m["bic"], "caic": m["caic"], "converged": m["converged"],
                    "elapsed_time": m["elapsed_time"],
                    "best_factor_n_factors": m["best_factor_n_factors"],
                    "best_factor_bic": m["best_factor_bic"],
                } for family, m in dense.items()},
                os.path.join(pc.EXACT_RESULTS_DIR, "dense_comparison_summary.json"),
            )
        print(f"\nStep 2 finished in {(time.time() - step_start) / 60:.1f} min "
              f"(this session). Results in {pc.EXACT_RESULTS_DIR}")
    finally:
        pc.stop_logging()


if __name__ == "__main__":
    main()
