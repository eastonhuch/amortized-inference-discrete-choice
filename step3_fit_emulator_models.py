"""
Step 3: emulator models.

Dense random effects (n_factors_rand = k) with time-specific factors
n_factors_time in EMULATOR_TIME_FACTORS_LIST. For each n_factors_time:
  1. the Gumbel emulators (gumbel-large, gumbel) are fit, warm-started from the
     exact dense mixed logit (first n_factors_time) or from their own previous
     solution;
  2. competitive refinement: the Gumbel emulator with the lower log-likelihood
     is refit from the other one's solution and keeps the refit if it improves;
  3. the probit emulator is fit, warm-started from the exact dense mixed probit
     or its own previous solution.

Final models are saved to <results root>/emulator_models/ as
{emulator}_dense_tf{n}.json, followed right away by their coefficient tables.
The initial and refit stages of the Gumbel models are checkpointed in
emulator_models/intermediate/, so an interruption loses at most one fit.

Requires the dense exact models from step 2.

    python step3_fit_emulator_models.py
    python step3_fit_emulator_models.py --no-coef-tables
"""
import argparse
import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import pipeline_common as pc
from analysis import (
    empty_cache,
    fit_single_model_with_emulator,
    get_param_names_from_kwargs,
    get_params_from_summary,
    result_dict_to_summary,
    save_model_summary,
)


def plot_emulator_weights(emulators: dict, path: str) -> None:
    fig, ax = plt.subplots(1, len(emulators), figsize=(2 * len(emulators) + 1, 2.5), squeeze=False)
    for i, (name, emulator) in enumerate(emulators.items()):
        weights = torch.cat([p.detach().flatten() for p in emulator.parameters()])
        ax[0, i].hist(weights.cpu().numpy(), bins=np.linspace(-3, 3, 100))
        ax[0, i].set_title(name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


class EmulatorFitter:
    def __init__(self, ctx: pc.AnalysisContext, emulators: dict, coef_tables: bool):
        self.ctx = ctx
        self.emulators = emulators
        self.coef_tables = coef_tables
        self.fit_order = [name for name in pc.emulators_to_fit() if name in emulators]
        self.gumbels = [name for name in pc.GUMBEL_EMULATORS if name in self.fit_order]
        self.tf_list = list(pc.EMULATOR_TIME_FACTORS_LIST)
        self.n_models = len(self.fit_order) * len(self.tf_list)
        self.model_count = 0

        # Exact dense models used for the first warm start
        self.exact_params = {}
        for family in {pc.emulator_family(name) for name in self.fit_order}:
            path = pc.dense_model_path(family)
            summary = pc.require_summary(path, "Run step2_fit_exact_models.py first.")
            self.exact_params[family] = (get_params_from_summary(summary, ctx.device), path)

    def model_kwargs(self, name: str) -> dict:
        # The model family is irrelevant when an emulator is used
        return self.ctx.model_kwargs("logit", include_uniquenesses_rand=False,
                                     emulator=self.emulators[name])

    def warm_start(self, name: str, tf_index: int):
        """(params, n_factors_time, description) for the first fit of `name` at tf_index."""
        if tf_index == 0:
            params, path = self.exact_params[pc.emulator_family(name)]
            return params.clone(), -1, os.path.basename(path)
        prev_tf = self.tf_list[tf_index - 1]
        path = pc.emulator_model_path(name, prev_tf)
        summary = pc.require_summary(path, "The previous time-factor model is missing.")
        return get_params_from_summary(summary, self.ctx.device), prev_tf, os.path.basename(path)

    def fit(self, name: str, tf: int, start_params, start_tf: int, source: str) -> dict:
        """Fit one emulator model and return its summary (not yet saved)."""
        print(f"  Warm-starting from {source} ({len(start_params)} parameters)")
        model_kwargs = self.model_kwargs(name)
        start = time.time()
        fit_result = fit_single_model_with_emulator(
            n_factors_rand=self.ctx.k_rand,
            n_factors_time=tf,
            batch_data_dicts=self.ctx.batches(pc.N_BATCHES_EMULATOR_FIT),
            start_param=start_params,
            start_param_n_factors_rand=self.ctx.k_rand,
            start_param_n_factors_time=start_tf,
            model_kwargs=model_kwargs,
            n_households=self.ctx.n_households,
            device=self.ctx.device,
            n_random_restarts=pc.N_RANDOM_RESTARTS_EMULATOR,
            restart_noise_std=1.0,
            early_cutoff=pc.EMULATOR_EARLY_CUTOFF,
        )
        elapsed = time.time() - start
        summary = result_dict_to_summary(
            fit_result,
            model_name=f"{name}_dense_tf{tf}",
            param_names=get_param_names_from_kwargs(
                model_kwargs, self.ctx.k_rand, tf, self.ctx.intercept_names, self.ctx.numeric_names
            ),
        )
        summary["elapsed_time"] = float(elapsed)
        summary["config"]["emulator_name"] = name
        summary["config"]["k_rand"] = int(self.ctx.k_rand)
        summary["warm_start"] = {"source": source, "n_factors_time": int(start_tf)}
        print(f"  Completed in {elapsed:.1f}s | Log-likelihood: {summary['log_likelihood']:.2f} "
              f"| Converged: {summary['converged']}")
        del fit_result
        gc.collect()
        empty_cache(self.ctx.device)
        return summary

    def fit_or_load(self, path: str, name: str, tf: int, start_fn) -> dict:
        """Load the summary at `path`, or fit the model (start_fn() gives the warm start)."""
        summary = pc.load_summary_if_exists(path)
        if summary is not None:
            print(f"  Loaded {path}")
            return summary
        summary = self.fit(name, tf, *start_fn())
        save_model_summary(summary, path)
        return summary

    def ensure_coef_table(self, name: str, tf: int) -> None:
        if self.coef_tables:
            pc.ensure_coef_table(
                self.ctx, pc.emulator_model_path(name, tf), pc.emulator_coef_table_path(name, tf),
                self.model_kwargs(name),
                os.path.join(pc.EMULATOR_RESULTS_DIR, "coef_table_timing_summary.json"),
                {"emulator": name, "n_factors_time": tf},
            )

    # ------------------------------------------------------------------
    # Gumbel emulators with competitive refinement
    # ------------------------------------------------------------------

    def gumbel_step(self, tf_index: int) -> None:
        tf = self.tf_list[tf_index]
        final_paths = {name: pc.emulator_model_path(name, tf) for name in self.gumbels}
        if all(os.path.exists(path) for path in final_paths.values()):
            for name in self.gumbels:
                self.model_count += 1
                print(f"\n[{self.model_count}/{self.n_models}] {name} (dense, tf={tf}): "
                      f"loaded {final_paths[name]}")
                self.ensure_coef_table(name, tf)
            return

        # Phase 1: initial fits
        initial = {}
        for name in self.gumbels:
            self.model_count += 1
            print(f"\n[{self.model_count}/{self.n_models}] {name} (dense, tf={tf}) - initial fit")
            initial[name] = self.fit_or_load(
                pc.emulator_intermediate_path(name, tf, "initial"), name, tf,
                lambda name=name: self.warm_start(name, tf_index),
            )

        final = {name: (initial[name], {"was_refitted": False}) for name in self.gumbels}

        # Phase 2: competitive refinement
        if len(self.gumbels) == 2:
            ll = {name: initial[name]["log_likelihood"] for name in self.gumbels}
            winner = "gumbel-large" if ll["gumbel-large"] > ll["gumbel"] else "gumbel"
            loser = "gumbel" if winner == "gumbel-large" else "gumbel-large"
            print(f"\n  Competitive refinement: gumbel-large LL={ll['gumbel-large']:.4f}, "
                  f"gumbel LL={ll['gumbel']:.4f} -> refitting {loser} from {winner}'s solution")
            refit = self.fit_or_load(
                pc.emulator_intermediate_path(loser, tf, "refit"), loser, tf,
                lambda: (get_params_from_summary(initial[winner], self.ctx.device), tf,
                         f"{winner} solution (tf={tf})"),
            )
            gain = refit["log_likelihood"] - ll[loser]
            if gain > 0:
                print(f"  Improvement: {gain:.4f} -> using the refitted {loser} result")
                refit = dict(refit, elapsed_time=initial[loser]["elapsed_time"] + refit["elapsed_time"])
                final[loser] = (refit, {"was_refitted": True, "refit_source": winner,
                                        "original_loglike": float(ll[loser])})
            else:
                print(f"  No improvement ({gain:.4f}) -> keeping the original {loser} result")

        # Phase 3: save final results, then coefficient tables
        for name in self.gumbels:
            summary, config_updates = final[name]
            save_model_summary(
                pc.mark_as_final(summary, f"{name}_dense_tf{tf}", **config_updates),
                final_paths[name],
            )
            refit_str = " (refitted)" if config_updates["was_refitted"] else ""
            print(f"  Saved {name}{refit_str}: LL={summary['log_likelihood']:.2f}, "
                  f"BIC={summary['bic']:.2f}")
        for name in self.gumbels:
            self.ensure_coef_table(name, tf)

    # ------------------------------------------------------------------
    # Probit emulator
    # ------------------------------------------------------------------

    def probit_step(self, tf_index: int) -> None:
        tf = self.tf_list[tf_index]
        self.model_count += 1
        print(f"\n[{self.model_count}/{self.n_models}] probit (dense, tf={tf})")
        path = pc.emulator_model_path("probit", tf)
        if pc.load_summary_if_exists(path) is None:
            summary = self.fit("probit", tf, *self.warm_start("probit", tf_index))
            save_model_summary(pc.mark_as_final(summary, f"probit_dense_tf{tf}", was_refitted=False), path)
        else:
            print(f"  Loaded {path}")
        self.ensure_coef_table("probit", tf)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def collect(self) -> dict:
        """Metrics of all final models on disk (None where missing)."""
        results = {}
        for name in self.fit_order:
            summaries = [pc.load_summary_if_exists(pc.emulator_model_path(name, tf)) for tf in self.tf_list]
            results[name] = summaries
        return results

    def save_comparison_summary(self) -> None:
        collected = self.collect()
        complete = all(s is not None for summaries in collected.values() for s in summaries)

        pc.plot_emulator_bic(
            {name: [None if s is None else s["bic"] for s in summaries]
             for name, summaries in collected.items()},
            self.tf_list,
            os.path.join(pc.EMULATOR_RESULTS_DIR, "emulator_progress_bic.pdf"),
            title="Emulator Model Comparison (progress)",
        )
        if not complete:
            return

        results = {}
        best = None
        for name, summaries in collected.items():
            results[name] = {
                "log_likelihoods": [float(s["log_likelihood"]) for s in summaries],
                "aics": [float(s["aic"]) for s in summaries],
                "bics": [float(s["bic"]) for s in summaries],
                "caics": [float(s["caic"]) for s in summaries],
                "times": [float(s.get("elapsed_time", float("nan"))) for s in summaries],
                "n_params": [int(s["n_params"]) for s in summaries],
                "was_refitted": [bool(s["config"].get("was_refitted", False)) for s in summaries],
            }
            for tf, s in zip(self.tf_list, summaries):
                if best is None or s["bic"] < best["bic"]:
                    best = {"emulator": name, "n_factors_time": int(tf), "bic": float(s["bic"])}

        refit_paths = [pc.emulator_intermediate_path(name, tf, "refit")
                       for name in self.gumbels for tf in self.tf_list]
        fit_times = [s["elapsed_time"] for s in collected.get("probit", [])]
        for name in self.gumbels:
            for tf in self.tf_list:
                for stage in ("initial", "refit"):
                    stage_summary = pc.load_summary_if_exists(pc.emulator_intermediate_path(name, tf, stage))
                    if stage_summary is not None:
                        fit_times.append(stage_summary["elapsed_time"])

        summary = {
            "emulator_mod_time_n_factors_list": [int(tf) for tf in self.tf_list],
            "k_rand": int(self.ctx.k_rand),
            "emulator_fit_order": self.fit_order,
            "competitive_refinement_enabled": len(self.gumbels) == 2,
            "results": results,
            "best_model": best,
            "total_time": float(sum(fit_times)),
            "refit_count": int(sum(os.path.exists(p) for p in refit_paths)),
        }
        save_model_summary(summary, os.path.join(pc.EMULATOR_RESULTS_DIR, "comparison_summary.json"))

        pc.banner("EMULATOR MODELS COMPLETE")
        for name in self.fit_order:
            print(f"\n{name} (dense):")
            r = results[name]
            for j, tf in enumerate(self.tf_list):
                star = " *" if r["was_refitted"][j] else ""
                print(f"  tf={tf}: LL={r['log_likelihoods'][j]:.2f}, BIC={r['bics'][j]:.2f}, "
                      f"n_params={r['n_params'][j]}, time={r['times'][j]:.1f}s{star}")
        if len(self.gumbels) == 2:
            print("\n  (* = result improved by competitive refinement)")
        print(f"\nBest by BIC: {best['emulator']} (dense) with tf={best['n_factors_time']}, "
              f"BIC={best['bic']:.2f}")
        print(f"Total fitting time: {summary['total_time'] / 60:.1f} min; refits: {summary['refit_count']}")

    def run(self) -> None:
        pc.banner("FITTING EMULATOR MODELS (dense covariance, competitive refinement)")
        print(f"Emulator fitting order: {self.fit_order}")
        print(f"Time factors: {self.tf_list}")
        print(f"Competitive refinement enabled: {len(self.gumbels) == 2}")
        for tf_index, tf in enumerate(self.tf_list):
            pc.banner(f"n_factors_time = {tf}")
            if self.gumbels:
                self.gumbel_step(tf_index)
            if "probit" in self.fit_order:
                self.probit_step(tf_index)
            self.save_comparison_summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 3: emulator models")
    parser.add_argument("--no-coef-tables", action="store_true",
                        help="skip coefficient tables (a later rerun adds the missing ones)")
    args = parser.parse_args(argv)

    pc.start_logging("step3_fit_emulator_models")
    try:
        ctx = pc.AnalysisContext()
        pc.check_results_dir(pc.EMULATOR_RESULTS_DIR, ctx.metadata)
        emulators = pc.load_emulators(ctx.device)
        plot_emulator_weights(emulators, os.path.join(pc.EMULATOR_RESULTS_DIR, "emulator_weights.pdf"))
        step_start = time.time()
        EmulatorFitter(ctx, emulators, coef_tables=not args.no_coef_tables).run()
        print(f"\nStep 3 finished in {(time.time() - step_start) / 60:.1f} min (this session). "
              f"Results in {pc.EMULATOR_RESULTS_DIR}")
    finally:
        pc.stop_logging()


if __name__ == "__main__":
    main()
