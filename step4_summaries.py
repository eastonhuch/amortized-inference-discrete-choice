"""
Step 4: summaries: model comparisons, tables, and emulator plots.

Reads the saved models of steps 2, 3, and 3b and writes:
  emulator_models/emulator_model_comparison_bic.pdf   BIC by time factors, with exact baselines
  emulator_models/analysis-results.csv                emulator model table
  summaries/exact_models.csv                          exact model table
  summaries/lr_tests.csv                              likelihood-ratio tests (comparisons 2-4)
  summaries/cross_evaluation.csv                      log-likelihoods of each method at each
                                                      solution (comparison 6)
  summaries/model_comparisons.txt                     the printed report
  summaries/dense_time_*                              dense time-covariance models (step 3b; see
                                                      dense_time_summaries.py)

Cross-evaluations are cached in summaries/cross_evaluation_cache.json (keyed
by the parameter values of each solution), so rerunning only evaluates what is
new or has been refit. Missing coefficient tables are not created here; rerun
step 2 or 3 for that.

    python step4_summaries.py
    python step4_summaries.py --skip-cross-eval
"""
import argparse
import os

import pandas as pd
from scipy.stats import chi2

import dense_time_summaries
import pipeline_common as pc
from analysis import (
    compute_household_loglikes_batched,
    get_params_from_summary,
    load_model_summary,
    save_model_summary,
)

LABELS = {"gumbel-large": "Gumbel-large", "gumbel": "Gumbel", "probit": "Probit"}
EXACT_LABELS = {"logit": "Exact MNL", "probit": "Exact MP"}


# =============================================================================
# Loading results
# =============================================================================

def load_exact_results() -> tuple:
    """Rows for all saved exact models, and the dense summaries by family."""
    rows, dense = [], {}
    for family in ["probit", "logit"]:
        paths = [(str(r), pc.factor_model_path(family, r)) for r in pc.N_FACTORS_RAND_LIST]
        paths.append(("dense", pc.dense_model_path(family)))
        for n_factors, path in paths:
            summary = pc.load_summary_if_exists(path)
            if summary is None:
                continue
            rows.append({
                "family": family, "n_factors": n_factors, **pc.summary_metrics(summary),
                "coef_table": os.path.exists(pc.exact_coef_table_path(path)),
                "file": os.path.basename(path),
            })
            if n_factors == "dense":
                dense[family] = summary
    return pd.DataFrame(rows), dense


def load_emulator_results(tf_list: list) -> dict:
    """{emulator: [summary or None for each n_factors_time]} for emulators with any results."""
    results = {}
    for name in pc.emulators_to_fit():
        summaries = [pc.load_summary_if_exists(pc.emulator_model_path(name, tf)) for tf in tf_list]
        if any(s is not None for s in summaries):
            results[name] = summaries
    return results


# =============================================================================
# Tables and plots
# =============================================================================

def emulator_results_table(emulator_results: dict, tf_list: list, k_rand: int) -> pd.DataFrame:
    rows = []
    for name, summaries in emulator_results.items():
        for tf, s in zip(tf_list, summaries):
            if s is None:
                continue
            m = pc.summary_metrics(s)
            rows.append({
                "emulator": name, "k_rand": k_rand, "n_factors_time": tf,
                "n_params": m["n_params"], "loglike": m["loglike"],
                "neg2_loglike": -2.0 * m["loglike"], "aic": m["aic"], "bic": m["bic"],
                "caic": m["caic"], "time_s": m["elapsed_time"],
                "was_refitted": bool(s["config"].get("was_refitted", False)),
            })
    return pd.DataFrame(rows)


def lr_test(ll_full: float, ll_reduced: float, df: int) -> tuple:
    """LR statistic and chi-square p-value (NaN when df <= 0)."""
    lr_stat = 2.0 * (ll_full - ll_reduced)
    p_value = float(chi2.sf(lr_stat, df)) if df > 0 else float("nan")
    return lr_stat, p_value


def print_lr_row(name: str, lr_stat: float, df: int, p_value: float) -> None:
    print(f"  {name:<40} LR = {lr_stat:>10.4f}, df = {df:>3}, p = {p_value:.4e}")


def model_comparisons(emulator_results: dict, dense: dict, tf_list: list) -> pd.DataFrame:
    """Prints model comparisons 1-5. Returns the LR tests as a table."""
    complete = {name: s for name, s in emulator_results.items() if all(x is not None for x in s)}
    for name in set(emulator_results) - set(complete):
        missing = [tf for tf, s in zip(tf_list, emulator_results[name]) if s is None]
        print(f"NOTE: {name} is missing n_factors_time={missing}; it is left out of the comparisons.")
    exact = {family: (float(s["log_likelihood"]), int(s["n_params"])) for family, s in dense.items()}
    usable = {name: s for name, s in complete.items() if pc.emulator_family(name) in exact}

    def ll_np(name, idx):
        s = usable[name][idx]
        return float(s["log_likelihood"]), int(s["n_params"])

    lr_rows = []

    pc.banner("COMPARISON 1: Log-likelihood comparison (Emulators vs Exact Models)")
    for family, (ll, _) in exact.items():
        print(f"{EXACT_LABELS[family] + ' (dense):':<28}LL = {ll:.4f}")
    for name in usable:
        exact_ll = exact[pc.emulator_family(name)][0]
        print(f"\n{LABELS[name]} emulator (dense):")
        for idx, tf in enumerate(tf_list):
            ll, n_params = ll_np(name, idx)
            print(f"  time={tf:>2}: LL = {ll:.4f}, params = {n_params}, "
                  f"diff from {EXACT_LABELS[pc.emulator_family(name)]}: {ll - exact_ll:.4f}")

    pc.banner("COMPARISON 2: Sequential LR tests (time=k vs time=k-1)")
    for name in usable:
        print(f"\n{LABELS[name]} emulator (dense):")
        for idx in range(1, len(tf_list)):
            (ll_f, np_f), (ll_r, np_r) = ll_np(name, idx), ll_np(name, idx - 1)
            lr_stat, p = lr_test(ll_f, ll_r, np_f - np_r)
            print_lr_row(f"time={tf_list[idx]} vs time={tf_list[idx - 1]}:", lr_stat, np_f - np_r, p)
            lr_rows.append({"comparison": "sequential", "emulator": name,
                            "full": f"time={tf_list[idx]}", "reduced": f"time={tf_list[idx - 1]}",
                            "lr_stat": lr_stat, "df": np_f - np_r, "p_value": p})

    pc.banner("COMPARISON 3: LR tests - Emulator (time=k) vs Exact Model")
    for name in usable:
        family = pc.emulator_family(name)
        ll_exact, np_exact = exact[family]
        print(f"\n{LABELS[name]} emulator (dense) vs {EXACT_LABELS[family]} (dense):")
        for idx, tf in enumerate(tf_list):
            ll_f, np_f = ll_np(name, idx)
            lr_stat, p = lr_test(ll_f, ll_exact, np_f - np_exact)
            print_lr_row(f"time={tf} vs {EXACT_LABELS[family]}:", lr_stat, np_f - np_exact, p)
            lr_rows.append({"comparison": "vs_exact", "emulator": name, "full": f"time={tf}",
                            "reduced": EXACT_LABELS[family], "lr_stat": lr_stat,
                            "df": np_f - np_exact, "p_value": p})

    if -1 in tf_list:
        pc.banner("COMPARISON 4: LR tests - Emulator (time=k) vs Emulator (time=-1)")
        idx_tm1 = tf_list.index(-1)
        for name in usable:
            ll_r, np_r = ll_np(name, idx_tm1)
            print(f"\n{LABELS[name]} emulator (dense):")
            for idx, tf in enumerate(tf_list):
                if tf == -1:
                    continue
                ll_f, np_f = ll_np(name, idx)
                lr_stat, p = lr_test(ll_f, ll_r, np_f - np_r)
                print_lr_row(f"time={tf} vs time=-1:", lr_stat, np_f - np_r, p)
                lr_rows.append({"comparison": "vs_time_minus_1", "emulator": name,
                                "full": f"time={tf}", "reduced": "time=-1", "lr_stat": lr_stat,
                                "df": np_f - np_r, "p_value": p})

    if len(exact) == 2:
        pc.banner("COMPARISON 5: Exact Models - Mixed Logit vs Mixed Probit (both dense)")
        (ll_l, np_l), (ll_p, np_p) = exact["logit"], exact["probit"]
        print(f"Mixed Logit (dense):  LL = {ll_l:.4f}, params = {np_l}")
        print(f"Mixed Probit (dense): LL = {ll_p:.4f}, params = {np_p}")
        print(f"LL difference (Probit - Logit): {ll_p - ll_l:.4f}")

    if usable:
        pc.banner("SUMMARY TABLE: All Emulator Models (dense)")
        print(f"{'Emulator':<15} {'time':<6} {'LL':>12} {'params':>8} {'vs Exact':>12} {'vs time=-1':>12}")
        print("-" * 70)
        for name in usable:
            exact_ll = exact[pc.emulator_family(name)][0]
            ll_tm1 = ll_np(name, tf_list.index(-1))[0] if -1 in tf_list else None
            for idx, tf in enumerate(tf_list):
                ll, n_params = ll_np(name, idx)
                diff_tm1 = "-" if (tf == -1 or ll_tm1 is None) else f"{ll - ll_tm1:>12.4f}"
                print(f"{LABELS[name]:<15} {tf:<6} {ll:>12.4f} {n_params:>8} "
                      f"{ll - exact_ll:>12.4f} {diff_tm1:>12}")
            print()
        print("Note: 'vs Exact' is the LL difference from the exact dense MNL (Gumbel "
              "emulators) or mixed probit (probit emulator);")
        print("      'vs time=-1' is the LL difference from the same emulator's time=-1 model.")
    return pd.DataFrame(lr_rows)


# =============================================================================
# Cross-evaluation (comparison 6)
# =============================================================================

class CrossEvaluator:
    """Household log-likelihoods of one method at another method's solution, cached on disk."""

    def __init__(self, ctx: pc.AnalysisContext, emulators: dict):
        self.ctx = ctx
        self.emulators = emulators
        self.cache_path = os.path.join(pc.SUMMARY_DIR, "cross_evaluation_cache.json")
        self.cache = {}
        if os.path.exists(self.cache_path):
            self.cache = load_model_summary(self.cache_path)
        self.rows = []

    def loglike(self, family: str, evaluator: str, solution_path: str, n_factors_time: int) -> float:
        summary = pc.require_summary(solution_path, "Run steps 2 and 3 first.")
        key = "|".join([evaluator, family, f"tf={n_factors_time}", os.path.basename(solution_path),
                        pc.fingerprint_values(summary["param_values"])])
        if key not in self.cache:
            print(f"  Evaluating {evaluator} at {os.path.basename(solution_path)} ...")
            loglikes = compute_household_loglikes_batched(
                batch_data_dicts=self.ctx.batches(pc.N_BATCHES_EVALUATION),
                params=get_params_from_summary(summary, self.ctx.device),
                model_kwargs=self.ctx.model_kwargs(family, include_uniquenesses_rand=False),
                n_factors_rand=self.ctx.k_rand,
                n_factors_time=n_factors_time,
                emulator=None if evaluator == "exact" else self.emulators[evaluator],
                device=self.ctx.device,
            )
            self.cache[key] = float(loglikes.sum())
            save_model_summary(self.cache, self.cache_path)
        value = self.cache[key]
        self.rows.append({"family": family, "evaluator": evaluator,
                          "solution": os.path.basename(solution_path),
                          "n_factors_time": n_factors_time, "loglike": value})
        return value

    def matrix(self, family: str, evaluators: list, solutions: dict, n_factors_time: int) -> pd.DataFrame:
        """Rows: evaluation method; columns: solution."""
        return pd.DataFrame({
            label: {ev: self.loglike(family, ev, path, n_factors_time) for ev in evaluators}
            for label, path in solutions.items()
        })


def _label(evaluator: str, family: str) -> str:
    return EXACT_LABELS[family] if evaluator == "exact" else LABELS[evaluator]


def _print_matrix(title: str, matrix: pd.DataFrame, family: str) -> None:
    shown = matrix.rename(index=lambda ev: _label(ev, family))
    print(f"\n{title}")
    print(shown.to_string(float_format=lambda v: f"{v:.4f}"))


def cross_evaluation(ctx: pc.AnalysisContext, emulators: dict, tf_list: list) -> pd.DataFrame:
    pc.banner("COMPARISON 6: Cross-evaluation of log-likelihoods at different solutions")
    print("Log-likelihood computed by each method (rows) at the optimum found by each")
    print("method (columns). With time_factors=-1, emulator and exact methods should")
    print("give identical log-likelihoods at any given parameter vector.")
    evaluator = CrossEvaluator(ctx, emulators)
    gumbels = [g for g in pc.GUMBEL_EMULATORS if g in emulators]
    own = {}

    if -1 in tf_list and gumbels:
        pc.banner("LOGIT MODELS: Cross-evaluation (dense)")
        solutions = {"Exact MNL": pc.dense_model_path("logit")}
        solutions.update({LABELS[g]: pc.emulator_model_path(g, -1) for g in gumbels})
        m = evaluator.matrix("logit", ["exact"] + gumbels, solutions, -1)
        _print_matrix("LOGIT (dense): rows = evaluation method, columns = solution", m, "logit")
        _print_matrix("LOGIT (dense): discrepancies from Exact MNL (should be ~0 for time_factors=-1)",
                      m.loc[gumbels] - m.loc["exact"], "logit")
        own["Exact MNL"] = m.loc["exact", "Exact MNL"]
        own.update({LABELS[g]: m.loc[g, LABELS[g]] for g in gumbels})

    if len(gumbels) == 2:
        pc.banner("GUMBEL vs GUMBEL-LARGE: Cross-evaluation for all time factors (dense)")
        overall = []
        for tf in tf_list:
            solutions = {f"{LABELS[g]} sol": pc.emulator_model_path(g, tf) for g in gumbels}
            m = evaluator.matrix("logit", gumbels, solutions, tf)
            _print_matrix(f"n_factors_time = {tf}", m, "logit")
            overall.append({
                "time_factors": tf,
                "Gumbel (own)": m.loc["gumbel", "Gumbel sol"],
                "Gumbel-large (own)": m.loc["gumbel-large", "Gumbel-large sol"],
                "Large - Gumbel at Gumbel sol": m.loc["gumbel-large", "Gumbel sol"] - m.loc["gumbel", "Gumbel sol"],
                "Large - Gumbel at Large sol": m.loc["gumbel-large", "Gumbel-large sol"] - m.loc["gumbel", "Gumbel-large sol"],
            })
        overall = pd.DataFrame(overall).set_index("time_factors")
        overall["Own-solution difference"] = overall["Gumbel-large (own)"] - overall["Gumbel (own)"]
        print("\nGUMBEL vs GUMBEL-LARGE: overall summary (dense)")
        print(overall.to_string(float_format=lambda v: f"{v:.4f}"))

    if -1 in tf_list and "probit" in emulators:
        pc.banner("PROBIT MODELS: Cross-evaluation (dense)")
        solutions = {"Exact MP": pc.dense_model_path("probit"),
                     "Probit Emu": pc.emulator_model_path("probit", -1)}
        m = evaluator.matrix("probit", ["exact", "probit"], solutions, -1)
        _print_matrix("PROBIT (dense): rows = evaluation method, columns = solution", m, "probit")
        _print_matrix("PROBIT (dense): discrepancies from Exact MP (should be ~0 for time_factors=-1)",
                      m.loc[["probit"]] - m.loc["exact"], "probit")
        own["Exact MP"] = m.loc["exact", "Exact MP"]
        own["Probit emulator"] = m.loc["probit", "Probit Emu"]

    pc.banner("COMBINED SUMMARY: log-likelihood of each method at its own solution (dense)")
    for label, value in own.items():
        print(f"  {label:<18} {value:.4f}")
    return pd.DataFrame(evaluator.rows).drop_duplicates()


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 4: summaries and model comparisons")
    parser.add_argument("--skip-cross-eval", action="store_true",
                        help="skip comparison 6 (needs the emulators and the data)")
    args = parser.parse_args(argv)

    os.makedirs(pc.SUMMARY_DIR, exist_ok=True)
    pc.start_logging("step4_summaries", extra_path=os.path.join(pc.SUMMARY_DIR, "model_comparisons.txt"))
    try:
        ctx = pc.AnalysisContext()
        for results_dir in (pc.EXACT_RESULTS_DIR, pc.EMULATOR_RESULTS_DIR, pc.SUMMARY_DIR):
            pc.check_results_dir(results_dir, ctx.metadata)
        tf_list = list(pc.EMULATOR_TIME_FACTORS_LIST)

        exact_df, dense = load_exact_results()
        pc.banner("EXACT MODELS")
        print(exact_df.to_string(index=False) if len(exact_df) else "No exact models found.")
        exact_df.to_csv(os.path.join(pc.SUMMARY_DIR, "exact_models.csv"), index=False)

        emulator_results = load_emulator_results(tf_list)
        emulator_df = emulator_results_table(emulator_results, tf_list, ctx.k_rand)
        if len(emulator_df):
            pc.banner("EMULATOR MODELS")
            print(emulator_df.to_string(index=False))
            emulator_df.to_csv(os.path.join(pc.EMULATOR_RESULTS_DIR, "analysis-results.csv"), index=False)
            baselines = {}
            if "logit" in dense:
                baselines["MNL dense"] = float(dense["logit"]["bic"])
            if "probit" in dense:
                baselines["Mixed Probit dense"] = float(dense["probit"]["bic"])
            plot_path = os.path.join(pc.EMULATOR_RESULTS_DIR, "emulator_model_comparison_bic.pdf")
            pc.plot_emulator_bic(
                {name: [None if s is None else s["bic"] for s in summaries]
                 for name, summaries in emulator_results.items()},
                tf_list, plot_path, baselines=baselines,
            )
            print(f"\nPlot saved to {plot_path}")

        lr_df = model_comparisons(emulator_results, dense, tf_list)
        lr_df.to_csv(os.path.join(pc.SUMMARY_DIR, "lr_tests.csv"), index=False)

        if not args.skip_cross_eval:
            complete = [name for name, s in emulator_results.items() if all(x is not None for x in s)]
            if complete and dense:
                emulators = pc.load_emulators(ctx.device, names=complete)
                cross_df = cross_evaluation(ctx, emulators, tf_list)
                cross_df.to_csv(os.path.join(pc.SUMMARY_DIR, "cross_evaluation.csv"), index=False)
            else:
                print("\nSkipping cross-evaluation: emulator or dense exact models are incomplete.")

        missing_tables = [row["file"] for _, row in exact_df.iterrows() if not row["coef_table"]]
        missing_tables += [
            os.path.basename(pc.emulator_model_path(r["emulator"], r["n_factors_time"]))
            for _, r in emulator_df.iterrows()
            if not os.path.exists(pc.emulator_coef_table_path(r["emulator"], r["n_factors_time"]))
        ]
        if missing_tables:
            print(f"\nNOTE: {len(missing_tables)} models have no coefficient table yet "
                  f"(rerun step 2/3 without --no-coef-tables): {missing_tables}")
        print(f"\nSummaries written to {pc.SUMMARY_DIR} and {pc.EMULATOR_RESULTS_DIR}")
        # Dense time-covariance models (step 3b), reported to their own files
        dense_time_summaries.report(ctx, skip_cross_eval=args.skip_cross_eval)
    finally:
        pc.stop_logging()


if __name__ == "__main__":
    main()
