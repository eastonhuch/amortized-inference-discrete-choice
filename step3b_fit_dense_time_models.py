"""
Step 3b: dense time-specific covariance (run between steps 3 and 4).

Dense random effects (n_factors_rand = k) and a dense time-specific covariance
(n_factors_time = n_brand_forms - 1: every time uniqueness is anchored at 0, so
Omega = L L' with a full Cholesky factor), fit with the emulators trained on
dense covariance matrices (DENSE_EMULATOR_CHECKPOINTS). For each dense emulator:
  1. one fit from each step 3 solution in DENSE_TIME_START_FACTORS (the IID
     (tf=-1) and 7-factor solutions of the matching factor-trained emulator,
     e.g. probit -> probit_dense). Each starting point has exactly the
     covariance matrices of its source model;
  2. competitive refinement, as in step 3: the Gumbel emulator with the lower
     log-likelihood is refit from the other one's best solution;
  3. the fit with the highest log-likelihood is final.

The dense emulators have their own approximation error, so each fit also
records the dense emulator's log-likelihood at its starting point
(warm_start.start_loglike) next to the source model's own log-likelihood.

Final models are saved to <results root>/dense_time_models/ as
{emulator}_dense_tf{n}.json, followed right away by their coefficient tables.
Every fit is checkpointed in dense_time_models/intermediate/, so an
interruption loses at most one fit.

Requires the step 3 models and the dense emulator checkpoints.

    python step3b_fit_dense_time_models.py
    python step3b_fit_dense_time_models.py --no-coef-tables
"""
import argparse
import gc
import os
import time

import torch

import pipeline_common as pc
from analysis import (
    ParamLayout,
    compute_household_loglikes_batched,
    empty_cache,
    fit_single_model_with_emulator,
    get_param_names_from_kwargs,
    get_params_from_summary,
    map_params_to_larger_model,
    result_dict_to_summary,
    save_model_summary,
)


class DenseTimeFitter:
    def __init__(self, ctx: pc.AnalysisContext, emulators: dict, coef_tables: bool):
        self.ctx = ctx
        self.emulators = emulators
        self.coef_tables = coef_tables
        self.fit_order = [name for name in pc.dense_emulators_to_fit() if name in emulators]
        self.gumbels = [name for name in self.fit_order
                        if pc.dense_source_emulator(name) in pc.GUMBEL_EMULATORS]
        self.tf = ctx.n_brand_forms - 1
        self.start_tfs = list(pc.DENSE_TIME_START_FACTORS)
        bad = [tf for tf in self.start_tfs if not -1 <= tf < self.tf]
        if bad:
            raise ValueError(f"DENSE_TIME_START_FACTORS {bad} must lie in -1..{self.tf - 1}.")
        self.n_models = len(self.fit_order)
        self.model_count = 0

        # Step 3 solutions used as starting points
        self.sources = {}
        for name in self.fit_order:
            for start_tf in self.start_tfs:
                path = pc.emulator_model_path(pc.dense_source_emulator(name), start_tf)
                summary = pc.require_summary(path, "Run step3_fit_emulator_models.py first.")
                self.sources[(name, start_tf)] = (summary, path)

    def model_kwargs(self, name: str) -> dict:
        # The model family is irrelevant when an emulator is used
        return self.ctx.model_kwargs("logit", include_uniquenesses_rand=False,
                                     emulator=self.emulators[name])

    def layout(self, n_factors_time: int) -> ParamLayout:
        return ParamLayout(
            n_brand_forms=self.ctx.n_brand_forms, X_dim=self.ctx.X_dim,
            random_feature_indices=self.ctx.random_feature_indices,
            n_factors_rand=self.ctx.k_rand, include_uniquenesses_rand=False,
            anchored_rand=True, n_factors_time=n_factors_time, has_time_params=True,
        )

    def map_to_dense(self, params: torch.Tensor, n_factors_time: int) -> torch.Tensor:
        return map_params_to_larger_model(
            source_params=params,
            source_n_factors_rand=self.ctx.k_rand,
            target_n_factors_rand=self.ctx.k_rand,
            source_n_factors_time=n_factors_time,
            target_n_factors_time=self.tf if n_factors_time >= 0 else 0,
            n_brand_forms=self.ctx.n_brand_forms,
            n_fixed_features=self.ctx.n_fixed_features,
            random_feature_indices=self.ctx.random_feature_indices,
            include_uniquenesses_rand=False,
            has_emulator=True,
            device=self.ctx.device,
        )

    def start_from_source(self, name: str, start_tf: int):
        """(dense params, source description, source log-likelihood) for one starting point."""
        summary, path = self.sources[(name, start_tf)]
        source = get_params_from_summary(summary, self.ctx.device)
        params = self.map_to_dense(source, start_tf)
        if start_tf < 0:
            # Omega = I is fixed in the tf=-1 model. Going through tf=0 (all
            # log-uniquenesses 0) keeps Omega = I; mapping directly would start
            # the new diagonal loadings at exp(-2).
            params = self.map_to_dense(params, 0)

        # The starting point must reproduce the source model's covariance matrices
        src_layout, dense_layout = self.layout(start_tf), self.layout(self.tf)
        for what in ("Sigma_rand", "Sigma_time"):
            diff = (getattr(src_layout, what)(source) - getattr(dense_layout, what)(params)).abs().max().item()
            if diff > 1e-5:
                raise RuntimeError(f"Starting point for {name} from tf={start_tf}: {what} differs by {diff:.2e}")
        return params, os.path.basename(path), float(summary["log_likelihood"])

    def start_loglike(self, name: str, params: torch.Tensor) -> float:
        """Log-likelihood of the dense model at `params` under emulator `name`."""
        loglikes = compute_household_loglikes_batched(
            batch_data_dicts=self.ctx.batches(pc.N_BATCHES_EVALUATION),
            params=params,
            model_kwargs=self.model_kwargs(name),
            n_factors_rand=self.ctx.k_rand,
            n_factors_time=self.tf,
            emulator=self.emulators[name],
            device=self.ctx.device,
        )
        return float(loglikes.sum())

    def fit(self, name: str, params: torch.Tensor, warm_start: dict) -> dict:
        """Fit one dense model from `params` and return its summary (not yet saved)."""
        warm_start["start_loglike"] = self.start_loglike(name, params)
        print(f"  Warm-starting from {warm_start['source']} ({len(params)} parameters); "
              f"source LL={warm_start['source_loglike']:.2f}, "
              f"LL at start under {name}={warm_start['start_loglike']:.2f}")
        model_kwargs = self.model_kwargs(name)
        start = time.time()
        fit_result = fit_single_model_with_emulator(
            n_factors_rand=self.ctx.k_rand,
            n_factors_time=self.tf,
            batch_data_dicts=self.ctx.batches(pc.N_BATCHES_EMULATOR_FIT),
            start_param=params,
            start_param_n_factors_rand=self.ctx.k_rand,
            start_param_n_factors_time=self.tf,
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
            model_name=f"{name}_dense_tf{self.tf}",
            param_names=get_param_names_from_kwargs(
                model_kwargs, self.ctx.k_rand, self.tf, self.ctx.intercept_names, self.ctx.numeric_names
            ),
        )
        summary["elapsed_time"] = float(elapsed)
        summary["config"]["emulator_name"] = name
        summary["config"]["k_rand"] = int(self.ctx.k_rand)
        summary["warm_start"] = warm_start
        print(f"  Completed in {elapsed:.1f}s | Log-likelihood: {summary['log_likelihood']:.2f} "
              f"| Converged: {summary['converged']}")
        del fit_result
        gc.collect()
        empty_cache(self.ctx.device)
        return summary

    def fit_or_load(self, path: str, name: str, start_fn) -> dict:
        """Load the summary at `path`, or fit the model (start_fn() gives params and warm_start)."""
        summary = pc.load_summary_if_exists(path)
        if summary is not None:
            print(f"  Loaded {path}")
            return summary
        summary = self.fit(name, *start_fn())
        save_model_summary(summary, path)
        return summary

    def fit_starts(self, name: str) -> dict:
        """{stage: summary} for the fits from each step 3 starting point."""
        self.model_count += 1
        fits = {}
        for start_tf in self.start_tfs:
            stage = f"from_tf{start_tf}"
            print(f"\n[{self.model_count}/{self.n_models}] {name} (dense, tf={self.tf}) - start from tf={start_tf}")

            def start_fn(start_tf=start_tf):
                params, source, source_ll = self.start_from_source(name, start_tf)
                return params, {"source": source, "n_factors_time": int(start_tf), "source_loglike": source_ll}

            fits[stage] = self.fit_or_load(pc.dense_time_intermediate_path(name, self.tf, stage), name, start_fn)
        return fits

    @staticmethod
    def best(fits: dict) -> str:
        return max(fits, key=lambda stage: fits[stage]["log_likelihood"])

    def ensure_coef_table(self, name: str) -> None:
        if self.coef_tables:
            pc.ensure_coef_table(
                self.ctx, pc.dense_time_model_path(name, self.tf), pc.dense_time_coef_table_path(name, self.tf),
                self.model_kwargs(name),
                os.path.join(pc.dense_time_results_dir(), "coef_table_timing_summary.json"),
                {"emulator": name, "n_factors_time": self.tf},
            )

    def save_final(self, name: str, fits: dict, refit_info: dict) -> None:
        stage = self.best(fits)
        summary = fits[stage]
        save_model_summary(
            pc.mark_as_final(summary, f"{name}_dense_tf{self.tf}", selected_fit=stage,
                             fit_loglikes={s: float(f["log_likelihood"]) for s, f in fits.items()},
                             **refit_info),
            pc.dense_time_model_path(name, self.tf),
        )
        print(f"  Saved {name}: best fit {stage}, LL={summary['log_likelihood']:.2f}, BIC={summary['bic']:.2f}")

    # ------------------------------------------------------------------

    def run(self) -> None:
        pc.banner(f"FITTING DENSE TIME-COVARIANCE MODELS (n_factors_time={self.tf}, dense random effects)")
        print(f"Emulators: {self.fit_order}")
        print(f"Starting points (step 3 n_factors_time): {self.start_tfs}")
        print(f"Competitive refinement enabled: {len(self.gumbels) == 2}")

        done = {name: pc.load_summary_if_exists(pc.dense_time_model_path(name, self.tf)) is not None
                for name in self.fit_order}

        # Gumbel emulators with competitive refinement
        if self.gumbels and not all(done[name] for name in self.gumbels):
            fits = {name: self.fit_starts(name) for name in self.gumbels}
            refit_info = {name: {"was_refitted": False} for name in self.gumbels}
            if len(self.gumbels) == 2:
                best_ll = {name: fits[name][self.best(fits[name])]["log_likelihood"] for name in self.gumbels}
                winner, loser = sorted(self.gumbels, key=lambda name: best_ll[name], reverse=True)
                winner_summary = fits[winner][self.best(fits[winner])]
                print(f"\n  Competitive refinement: {winner} LL={best_ll[winner]:.4f}, "
                      f"{loser} LL={best_ll[loser]:.4f} -> refitting {loser} from {winner}'s solution")

                def start_fn():
                    return (get_params_from_summary(winner_summary, self.ctx.device),
                            {"source": f"{winner} solution (tf={self.tf})", "n_factors_time": int(self.tf),
                             "source_loglike": float(winner_summary["log_likelihood"])})

                refit = self.fit_or_load(pc.dense_time_intermediate_path(loser, self.tf, "refit"), loser, start_fn)
                gain = refit["log_likelihood"] - best_ll[loser]
                if gain > 0:
                    print(f"  Improvement: {gain:.4f} -> using the refitted {loser} result")
                    fits[loser]["refit"] = refit
                    refit_info[loser] = {"was_refitted": True, "refit_source": winner,
                                         "original_loglike": float(best_ll[loser])}
                else:
                    print(f"  No improvement ({gain:.4f}) -> keeping the original {loser} result")
            for name in self.gumbels:
                self.save_final(name, fits[name], refit_info[name])

        # Remaining emulators (probit)
        for name in self.fit_order:
            if name in self.gumbels or done[name]:
                continue
            self.save_final(name, self.fit_starts(name), {"was_refitted": False})

        for name in self.fit_order:
            if done[name]:
                print(f"\n{name}: loaded {pc.dense_time_model_path(name, self.tf)}")
            self.ensure_coef_table(name)
        self.save_comparison_summary()

    def save_comparison_summary(self) -> None:
        results = {}
        for name in self.fit_order:
            final = pc.require_summary(pc.dense_time_model_path(name, self.tf), "Fit the model first.")
            sources = {}
            for start_tf in self.start_tfs:
                summary, _ = self.sources[(name, start_tf)]
                sources[f"tf{start_tf}"] = {
                    "emulator": pc.dense_source_emulator(name),
                    "log_likelihood": float(summary["log_likelihood"]),
                    "bic": float(summary["bic"]),
                    "n_params": int(summary["n_params"]),
                }
            results[name] = {
                "log_likelihood": float(final["log_likelihood"]),
                "bic": float(final["bic"]),
                "n_params": int(final["n_params"]),
                "selected_fit": final["config"].get("selected_fit"),
                "was_refitted": bool(final["config"].get("was_refitted", False)),
                "fit_loglikes": final["config"].get("fit_loglikes"),
                "sources": sources,
            }
        save_model_summary(
            {"n_factors_time": int(self.tf), "k_rand": int(self.ctx.k_rand),
             "start_factors": [int(tf) for tf in self.start_tfs], "results": results},
            os.path.join(pc.dense_time_results_dir(), "comparison_summary.json"),
        )

        pc.banner("DENSE TIME-COVARIANCE MODELS COMPLETE")
        for name, r in results.items():
            print(f"\n{name} (dense, tf={self.tf}): LL={r['log_likelihood']:.2f}, BIC={r['bic']:.2f}, "
                  f"n_params={r['n_params']}, best fit: {r['selected_fit']}"
                  f"{' (refitted)' if r['was_refitted'] else ''}")
            for stage, ll in (r["fit_loglikes"] or {}).items():
                print(f"  {stage:<12} LL={ll:.2f}")
            for label, s in r["sources"].items():
                print(f"  step 3 {s['emulator']} {label}: LL={s['log_likelihood']:.2f}, "
                      f"BIC={s['bic']:.2f}, n_params={s['n_params']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 3b: dense time-specific covariance models")
    parser.add_argument("--no-coef-tables", action="store_true",
                        help="skip coefficient tables (a later rerun adds the missing ones)")
    args = parser.parse_args(argv)

    pc.start_logging("step3b_fit_dense_time_models")
    try:
        ctx = pc.AnalysisContext()
        pc.check_results_dir(pc.EMULATOR_RESULTS_DIR, ctx.metadata)
        pc.check_results_dir(pc.dense_time_results_dir(), ctx.metadata)
        emulators = pc.load_emulators(ctx.device, names=pc.dense_emulators_to_fit(),
                                      checkpoints=pc.DENSE_EMULATOR_CHECKPOINTS)
        step_start = time.time()
        DenseTimeFitter(ctx, emulators, coef_tables=not args.no_coef_tables).run()
        print(f"\nStep 3b finished in {(time.time() - step_start) / 60:.1f} min (this session). "
              f"Results in {pc.dense_time_results_dir()}")
    finally:
        pc.stop_logging()


if __name__ == "__main__":
    main()
