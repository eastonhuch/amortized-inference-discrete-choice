"""
Step 4 summaries of the dense time-covariance models (step 3b).

Called at the end of step4_summaries.py, after its other outputs are written;
the printed report goes to its own file, separate from model_comparisons.txt.
Writes:
  summaries/dense_time_models.txt              the printed report
  summaries/dense_time_models.csv              step 3 starting models, every dense fit, final models
  summaries/dense_time_lr_tests.csv            LR tests of the dense models against their starting models
  summaries/dense_time_cross_evaluation.csv    Gumbel cross-evaluation at the dense solutions
  dense_time_models/emulator_bic_with_dense.pdf
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

import pipeline_common as pc
from analysis import (
    compute_household_loglikes_batched,
    get_params_from_summary,
    load_model_summary,
    save_model_summary,
)


def _row(dense_name: str, model: str, emulator: str, n_factors_time: int, summary: dict,
         start_loglike=None) -> dict:
    m = pc.summary_metrics(summary)
    return {
        "dense_emulator": dense_name, "model": model, "emulator": emulator,
        "n_factors_time": n_factors_time, "n_params": m["n_params"], "loglike": m["loglike"],
        "neg2_loglike": -2.0 * m["loglike"], "aic": m["aic"], "bic": m["bic"],
        "start_loglike": start_loglike, "converged": m["converged"], "time_s": m["elapsed_time"],
    }


def _stages() -> list:
    return [f"from_tf{tf}" for tf in pc.DENSE_TIME_START_FACTORS] + ["refit"]


def model_table(finals: dict, tf: int) -> pd.DataFrame:
    rows = []
    for name, final in finals.items():
        source = pc.dense_source_emulator(name)
        for start_tf in pc.DENSE_TIME_START_FACTORS:
            summary = pc.load_summary_if_exists(pc.emulator_model_path(source, start_tf))
            if summary is not None:
                rows.append(_row(name, f"step 3 tf={start_tf}", source, start_tf, summary))
        for stage in _stages():
            summary = pc.load_summary_if_exists(pc.dense_time_intermediate_path(name, tf, stage))
            if summary is not None:
                rows.append(_row(name, f"dense {stage}", name, tf, summary,
                                 summary.get("warm_start", {}).get("start_loglike")))
        rows.append(_row(name, f"dense final ({final['config'].get('selected_fit')})", name, tf, final))
    return pd.DataFrame(rows)


def lr_tests(finals: dict, tf: int) -> pd.DataFrame:
    rows = []
    for name, final in finals.items():
        ll_full, np_full = float(final["log_likelihood"]), int(final["n_params"])
        source = pc.dense_source_emulator(name)
        for start_tf in pc.DENSE_TIME_START_FACTORS:
            reduced = pc.load_summary_if_exists(pc.emulator_model_path(source, start_tf))
            start = pc.load_summary_if_exists(pc.dense_time_intermediate_path(name, tf, f"from_tf{start_tf}"))
            if reduced is None:
                continue
            df = np_full - int(reduced["n_params"])
            for basis, ll_reduced in (
                (f"{source} optimum", float(reduced["log_likelihood"])),
                (f"{name} at {source} optimum",
                 None if start is None else start.get("warm_start", {}).get("start_loglike")),
            ):
                if ll_reduced is None:
                    continue
                lr_stat = 2.0 * (ll_full - ll_reduced)
                rows.append({
                    "dense_emulator": name, "full": f"dense tf={tf}", "reduced": f"tf={start_tf}",
                    "reduced_loglike_from": basis, "loglike_full": ll_full, "loglike_reduced": ll_reduced,
                    "lr_stat": lr_stat, "df": df,
                    "p_value": float(chi2.sf(lr_stat, df)) if df > 0 else float("nan"),
                })
    return pd.DataFrame(rows)


def plot_bic(finals: dict, tf: int, path: str) -> None:
    """BIC of the step 3 emulator models by time factors, with the dense models at tf = k_time."""
    tf_list = list(pc.EMULATOR_TIME_FACTORS_LIST)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, final in finals.items():
        source = pc.dense_source_emulator(name)
        color = pc.EMULATOR_COLORS.get(source, "tab:gray")
        points = [(t, s["bic"]) for t in tf_list
                  for s in [pc.load_summary_if_exists(pc.emulator_model_path(source, t))] if s is not None]
        if points:
            ax.plot(*zip(*points), marker="o", linestyle="-", color=color, label=f"{source} (factor emulator)")
        ax.plot([tf], [final["bic"]], marker="*", markersize=14, linestyle="none", color=color,
                label=f"{name} (dense, tf={tf})")
    ax.legend(loc="best")
    ax.set_xlabel("Time-Specific Errors: # Factors")
    ax.set_ylabel("BIC")
    ax.set_title("Emulator Models by BIC, with Dense Time-Specific Covariance")
    ax.set_xticks(tf_list + [tf])
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def cross_evaluation(ctx: pc.AnalysisContext, finals: dict, tf: int) -> pd.DataFrame:
    """Log-likelihood of each dense Gumbel emulator (rows) at each dense Gumbel solution (columns)."""
    gumbels = [name for name in finals if pc.dense_source_emulator(name) in pc.GUMBEL_EMULATORS]
    if len(gumbels) < 2:
        print("\nSkipping cross-evaluation: needs both dense Gumbel models.")
        return pd.DataFrame()
    emulators = pc.load_emulators(ctx.device, names=gumbels, checkpoints=pc.DENSE_EMULATOR_CHECKPOINTS)
    cache_path = os.path.join(pc.SUMMARY_DIR, "dense_time_cross_evaluation_cache.json")
    cache = load_model_summary(cache_path) if os.path.exists(cache_path) else {}
    rows = []
    for evaluator in gumbels:
        for solution in gumbels:
            summary = finals[solution]
            path = pc.dense_time_model_path(solution, tf)
            key = "|".join([evaluator, os.path.basename(path), pc.fingerprint_values(summary["param_values"])])
            if key not in cache:
                print(f"  Evaluating {evaluator} at {os.path.basename(path)} ...")
                loglikes = compute_household_loglikes_batched(
                    batch_data_dicts=ctx.batches(pc.N_BATCHES_EVALUATION),
                    params=get_params_from_summary(summary, ctx.device),
                    model_kwargs=ctx.model_kwargs("logit", include_uniquenesses_rand=False),
                    n_factors_rand=ctx.k_rand,
                    n_factors_time=tf,
                    emulator=emulators[evaluator],
                    device=ctx.device,
                )
                cache[key] = float(loglikes.sum())
                save_model_summary(cache, cache_path)
            rows.append({"evaluator": evaluator, "solution": solution, "n_factors_time": tf,
                         "loglike": cache[key]})
    df = pd.DataFrame(rows)
    pc.banner("DENSE GUMBEL vs DENSE GUMBEL-LARGE: cross-evaluation (rows = evaluator, columns = solution)")
    print(df.pivot(index="evaluator", columns="solution", values="loglike")
          .to_string(float_format=lambda v: f"{v:.4f}"))
    return df


def report(ctx: pc.AnalysisContext, skip_cross_eval: bool = False) -> None:
    pc.stop_logging()
    pc.start_logging("step4_dense_time_summaries",
                     extra_path=os.path.join(pc.SUMMARY_DIR, "dense_time_models.txt"))
    tf = ctx.n_brand_forms - 1
    finals = {}
    for name in pc.dense_emulators_to_fit():
        summary = pc.load_summary_if_exists(pc.dense_time_model_path(name, tf))
        if summary is not None:
            finals[name] = summary

    pc.banner(f"DENSE TIME-SPECIFIC COVARIANCE MODELS (step 3b, n_factors_time={tf})")
    if not finals:
        print("No dense time-covariance models found; run step3b_fit_dense_time_models.py to add them.")
        return
    pc.check_results_dir(pc.dense_time_results_dir(), ctx.metadata)

    table = model_table(finals, tf)
    print(table.to_string(index=False))
    print("\nstart_loglike: log-likelihood of the dense model at the fit's starting point, under the "
          "dense emulator.\nThe step 3 rows use the factor-trained emulators, so their log-likelihoods "
          "include a different approximation error.")
    table.to_csv(os.path.join(pc.SUMMARY_DIR, "dense_time_models.csv"), index=False)

    pc.banner("LR TESTS: dense time covariance vs the step 3 starting models")
    lr = lr_tests(finals, tf)
    for _, r in lr.iterrows():
        print(f"  {r['dense_emulator']:<20} dense vs {r['reduced']:<6} (reduced LL: {r['reduced_loglike_from']:<34}) "
              f"LR = {r['lr_stat']:>10.4f}, df = {r['df']:>3}, p = {r['p_value']:.4e}")
    print("\nReduced LL from the factor emulator's optimum: the two models use different emulators.\n"
          "Reduced LL from the dense emulator at that optimum: same emulator, but the restricted model is "
          "not re-optimized\nunder it, so this LR statistic is an upper bound.")
    lr.to_csv(os.path.join(pc.SUMMARY_DIR, "dense_time_lr_tests.csv"), index=False)

    plot_path = os.path.join(pc.dense_time_results_dir(), "emulator_bic_with_dense.pdf")
    plot_bic(finals, tf, plot_path)
    print(f"\nPlot saved to {plot_path}")

    if not skip_cross_eval:
        cross = cross_evaluation(ctx, finals, tf)
        if len(cross):
            cross.to_csv(os.path.join(pc.SUMMARY_DIR, "dense_time_cross_evaluation.csv"), index=False)
    print(f"\nDense time-covariance summaries written to {pc.SUMMARY_DIR}")
