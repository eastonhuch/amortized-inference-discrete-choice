"""
Shared configuration and helpers for the analysis pipeline.

The case-study analysis consists of five scripts that are run in order:

    python step1_prepare_data.py          # data cache and household draws
    python step2_fit_exact_models.py      # mixed probit/logit: factor models, then dense
    python step3_fit_emulator_models.py   # dense emulator models over time factors
    python step3b_fit_dense_time_models.py  # dense time covariance with the dense emulators
    python step4_summaries.py             # plots, tables, LR tests, cross-evaluation

Resuming
--------
Every model is written to its JSON summary as soon as it is fit, and its
coefficient table is created right after (the joint covariance matrix is added
to the summary at the same time). Each script skips work whose results are
already on disk, so rerunning a script after an interruption continues from
the last saved model, warm-starting from the saved parameters exactly as an
uninterrupted run would. To refit a model, delete its JSON file (and the files
of any models that were warm-started from it).

Results directories record a fingerprint of the data, the random draws and the
settings that determine the fits (run_info.json). A script refuses to add to a
directory whose fingerprint does not match, so stale checkpoints are never
reused silently.

Settings are in the CONFIGURATION section below. The results root can also be
set with the environment variable ANALYSIS_RESULTS_ROOT (useful for a test run
on a household subsample, see HOUSEHOLD_SAMPLE_FRAC).
"""
import copy
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis import (  # noqa: E402
    PARAM_VERSION,
    add_covariance_to_summary,
    create_coefficient_table_from_summary,
    empty_cache,
    load_model_summary,
    save_model_summary,
    split_data_by_household,
)
from utils import get_device  # noqa: E402


# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Output locations --------------------------------------------------------
RESULTS_ROOT = os.environ.get("ANALYSIS_RESULTS_ROOT", "analysis-results")

# ---- Data (step 1) -----------------------------------------------------------
PANEL_FILE = "full_panel.csv"            # read from nielsen_shared.BASE_DIR
MIN_PURCHASES = 10                       # keep households with MIN..MAX purchases
MAX_PURCHASES = 30
HOUSEHOLD_SAMPLE_FRAC = None             # e.g. 0.1 for a quick test on 10% of households
HOUSEHOLD_SAMPLE_SEED = 42
NUMERIC_NAMES = [
    "loyalty: brand-form", "loyalty: brand", "loyalty: form",
    "coupon", "price", "promotion",
]
RANDOM_FEATURE_INDICES = [3, 4, 5]       # indices into NUMERIC_NAMES

# ---- Simulation draws --------------------------------------------------------
N_DRAWS = 200
RANDOM_SEED = 42

# ---- Household batches used for each task -------------------------------------
N_BATCHES_FACTOR_FIT = 3
N_BATCHES_DENSE_FIT = 10
N_BATCHES_EMULATOR_FIT = 200
N_BATCHES_COEF_TABLES = 1000
N_BATCHES_EVALUATION = 200

# ---- Exact (non-emulator) models (step 2) ---------------------------------------
N_FACTORS_RAND_LIST = [-1, 0, 1, 2, 3, 4, 5, 6]
N_RANDOM_RESTARTS = 19
RESTART_NOISE_STD = 1.0

# ---- Emulator models (step 3) ---------------------------------------------------
EMULATOR_CHECKPOINTS = {
    "gumbel-large": "./model-checkpoints/gumbel-large12.pt",
    "gumbel": "./model-checkpoints/gumbel12.pt",
    "probit": "./model-checkpoints/probit12.pt",
}
EMULATOR_FIT_ORDER = ["gumbel-large", "gumbel", "probit"]
EMULATOR_TIME_FACTORS_LIST = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
N_RANDOM_RESTARTS_EMULATOR = 0
EMULATOR_EARLY_CUTOFF = 200

EMULATOR_COLORS = {"probit": "tab:blue", "gumbel": "tab:orange", "gumbel-large": "tab:green"}

# ---- Dense time-specific covariance (step 3b) -------------------------------------
# Emulators trained on full-rank covariance matrices (fit-*-dense-analysis-emulator.ipynb)
DENSE_EMULATOR_CHECKPOINTS = {
    "gumbel-large_dense": "./model-checkpoints/gumbel-large_dense12.pt",
    "gumbel_dense": "./model-checkpoints/gumbel_dense12.pt",
    "probit_dense": "./model-checkpoints/probit_dense12.pt",
}
DENSE_EMULATOR_FIT_ORDER = ["gumbel-large_dense", "gumbel_dense", "probit_dense"]
DENSE_TIME_START_FACTORS = [-1, 7]   # step 3 solutions (n_factors_time) used as starting points


# =============================================================================
# Paths and file names
# =============================================================================

FAMILIES = {
    "probit": {"prefix": "mp", "label": "Mixed Probit"},
    "logit": {"prefix": "mnl", "label": "Mixed Logit"},
}
GUMBEL_EMULATORS = ["gumbel-large", "gumbel"]


def set_results_root(root: str) -> None:
    """Point all output directories at a new results root."""
    global RESULTS_ROOT, PREPARED_DIR, EXACT_RESULTS_DIR, EMULATOR_RESULTS_DIR
    global SUMMARY_DIR, LOG_DIR
    RESULTS_ROOT = root
    PREPARED_DIR = os.path.join(root, "prepared")
    EXACT_RESULTS_DIR = os.path.join(root, "mnl_models")
    EMULATOR_RESULTS_DIR = os.path.join(root, "emulator_models")
    SUMMARY_DIR = os.path.join(root, "summaries")
    LOG_DIR = os.path.join(root, "logs")


set_results_root(RESULTS_ROOT)


def factor_model_path(family: str, n_factors: int) -> str:
    return os.path.join(EXACT_RESULTS_DIR, f"{FAMILIES[family]['prefix']}_factors_{n_factors}.json")


def dense_model_path(family: str) -> str:
    return os.path.join(EXACT_RESULTS_DIR, f"{FAMILIES[family]['prefix']}_dense.json")


def exact_coef_table_path(summary_path: str) -> str:
    return summary_path[: -len(".json")] + "_coef_table.csv"


def emulator_model_path(name: str, n_factors_time: int) -> str:
    return os.path.join(EMULATOR_RESULTS_DIR, f"{name}_dense_tf{n_factors_time}.json")


def emulator_coef_table_path(name: str, n_factors_time: int) -> str:
    return os.path.join(EMULATOR_RESULTS_DIR, f"{name}_dense_tf{n_factors_time}_coefficients.csv")


def emulator_intermediate_path(name: str, n_factors_time: int, stage: str) -> str:
    """Checkpoints of the competitive-refinement stages ('initial', 'refit')."""
    return os.path.join(
        EMULATOR_RESULTS_DIR, "intermediate", f"{name}_dense_tf{n_factors_time}_{stage}.json"
    )


def emulator_family(name: str) -> str:
    """Exact model family whose dense fit warm-starts (and benchmarks) an emulator."""
    return "probit" if name == "probit" else "logit"


def emulators_to_fit() -> list:
    return [name for name in EMULATOR_FIT_ORDER if name in EMULATOR_CHECKPOINTS]


def dense_time_results_dir() -> str:
    return os.path.join(RESULTS_ROOT, "dense_time_models")


def dense_time_model_path(name: str, n_factors_time: int) -> str:
    return os.path.join(dense_time_results_dir(), f"{name}_dense_tf{n_factors_time}.json")


def dense_time_coef_table_path(name: str, n_factors_time: int) -> str:
    return os.path.join(dense_time_results_dir(), f"{name}_dense_tf{n_factors_time}_coefficients.csv")


def dense_time_intermediate_path(name: str, n_factors_time: int, stage: str) -> str:
    """Checkpoints of the individual fits ('from_tf-1', 'from_tf7', 'refit')."""
    return os.path.join(
        dense_time_results_dir(), "intermediate", f"{name}_dense_tf{n_factors_time}_{stage}.json"
    )


def dense_source_emulator(name: str) -> str:
    """Factor-trained emulator whose step 3 solutions warm-start a dense emulator."""
    return name[: -len("_dense")] if name.endswith("_dense") else name


def dense_emulators_to_fit() -> list:
    return [name for name in DENSE_EMULATOR_FIT_ORDER if name in DENSE_EMULATOR_CHECKPOINTS]


# =============================================================================
# Logging
# =============================================================================

class _Tee:
    """Write to several streams at once (console + log files)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return getattr(self.streams[0], "isatty", lambda: False)()


class _CarriageReturnFilter:
    """
    Writer for log files: applies carriage returns the way a terminal does, so a
    progress bar that redraws its line with '\r' leaves only its final state.
    """

    def __init__(self, stream):
        self.stream = stream
        self._pending = ""   # current, unfinished line

    def write(self, data):
        lines = (self._pending + data).split("\n")
        self._pending = lines.pop()
        for line in lines:
            self.stream.write(line.rstrip("\r").rsplit("\r", 1)[-1] + "\n")
        # Drop overwritten content of the unfinished line to bound memory
        cut = self._pending.rstrip("\r").rfind("\r")
        if cut >= 0:
            self._pending = self._pending[cut + 1:]
        return len(data)

    def finish(self):
        """Write the last unfinished line (e.g. a bar left open by a crash)."""
        remainder = self._pending.rstrip("\r").rsplit("\r", 1)[-1]
        if remainder.strip():
            self.stream.write(remainder + "\n")
        self._pending = ""
        self.stream.flush()

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


_log_files = []
_filters = []


def _log_writer(stream):
    """Terminals get the raw stream (live progress bars); files get the filter."""
    if getattr(stream, "isatty", lambda: False)():
        return stream
    writer = _CarriageReturnFilter(stream)
    _filters.append(writer)
    return writer


def start_logging(step_name: str, extra_path: str = None) -> str:
    """Copy stdout/stderr to a time-stamped log file (and optionally a report file)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{step_name}_{stamp}.log")
    _log_files.append(open(log_path, "a", buffering=1))
    if extra_path is not None:
        os.makedirs(os.path.dirname(extra_path), exist_ok=True)
        _log_files.append(open(extra_path, "w", buffering=1))
    sys.stdout = _Tee(_log_writer(sys.__stdout__), *[_log_writer(f) for f in _log_files])
    sys.stderr = _Tee(_log_writer(sys.__stderr__), *[_log_writer(f) for f in _log_files])
    print(f"[{stamp}] {step_name}: logging to {log_path}")
    return log_path


def stop_logging() -> None:
    while _filters:
        _filters.pop().finish()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    while _log_files:
        _log_files.pop().close()


def banner(text: str, char: str = "=", width: int = 70) -> None:
    print("\n" + char * width)
    print(text)
    print(char * width)


# =============================================================================
# Torch setup, draws and fingerprints
# =============================================================================

def setup_torch() -> torch.device:
    """Deterministic cuDNN settings and device selection."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = get_device()
    print(f"Device: {device}")
    return device


def make_std_normal_draws(n_households: int, n_draws: int, max_k: int, seed: int) -> torch.Tensor:
    """
    Standard normal draws for every household, shape (n_households, n_draws, max_k).

    Generated on the CPU with a dedicated generator, so every step reproduces
    exactly the same draws regardless of the device (the fingerprint stored in
    step 1 is checked by the later steps).
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(n_households, n_draws, max_k, generator=generator, dtype=torch.float32)


def fingerprint_tensors(tensors) -> str:
    """Short SHA-256 fingerprint of a sequence of tensors (values, dtypes and shapes)."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def fingerprint_values(values) -> str:
    """Short fingerprint of JSON-serializable values (e.g. a parameter vector)."""
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


# =============================================================================
# Prepared data (written by step 1)
# =============================================================================

def prepared_data_path() -> str:
    return os.path.join(PREPARED_DIR, "data.pt")


def prepared_metadata_path() -> str:
    return os.path.join(PREPARED_DIR, "metadata.json")


def save_prepared_data(data_dict: dict, metadata: dict) -> None:
    os.makedirs(PREPARED_DIR, exist_ok=True)
    tmp_path = prepared_data_path() + ".tmp"
    torch.save({"data_dict": data_dict}, tmp_path)
    os.replace(tmp_path, prepared_data_path())
    save_model_summary(metadata, prepared_metadata_path())   # atomic JSON write


def load_prepared_data():
    if not (os.path.exists(prepared_data_path()) and os.path.exists(prepared_metadata_path())):
        raise FileNotFoundError(
            f"No prepared data in {PREPARED_DIR}. Run step1_prepare_data.py first."
        )
    try:
        payload = torch.load(prepared_data_path(), map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        payload = torch.load(prepared_data_path(), map_location="cpu")
    return payload["data_dict"], load_model_summary(prepared_metadata_path())


def check_results_dir(results_dir: str, metadata: dict) -> None:
    """
    Make sure `results_dir` belongs to this data/draws/settings combination.
    Creates run_info.json on first use.
    """
    os.makedirs(results_dir, exist_ok=True)
    info_path = os.path.join(results_dir, "run_info.json")
    fingerprint = metadata["run_fingerprint"]
    if os.path.exists(info_path):
        stored = load_model_summary(info_path).get("run_fingerprint")
        if stored != fingerprint:
            raise RuntimeError(
                f"{results_dir} holds results for a different data set, draws or "
                f"settings (run fingerprint {stored}, current {fingerprint}). Move "
                "that directory away or set ANALYSIS_RESULTS_ROOT to a new location."
            )
        return
    existing = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if existing:
        raise RuntimeError(
            f"{results_dir} already contains {len(existing)} JSON files that were not "
            "written by this pipeline (no run_info.json). Move them elsewhere or set "
            "ANALYSIS_RESULTS_ROOT to a new location."
        )
    save_model_summary(
        {
            "run_fingerprint": fingerprint,
            "param_version": PARAM_VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "settings": metadata.get("fingerprint_inputs"),
        },
        info_path,
    )


# =============================================================================
# Analysis context: data, draws, batches, model kwargs
# =============================================================================

def _to_device(data_dict: dict, device) -> dict:
    out = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            out[key] = {k: v.to(device) for k, v in value.items()}
        else:
            out[key] = value
    return out


class AnalysisContext:
    """Prepared data on the device, the household draws, and cached batch splits."""

    def __init__(self, device=None):
        self.device = device if device is not None else setup_torch()
        data_dict, metadata = load_prepared_data()
        if metadata["n_draws"] != N_DRAWS or metadata["random_seed"] != RANDOM_SEED:
            raise RuntimeError(
                "N_DRAWS / RANDOM_SEED differ from the values used in step 1 "
                f"({metadata['n_draws']}, {metadata['random_seed']}). Rerun "
                "step1_prepare_data.py --force (with a new results root)."
            )
        self.metadata = metadata
        self.data_dict = _to_device(data_dict, self.device)
        self.n_households = int(metadata["n_households"])
        self.n_brand_forms = int(metadata["n_brand_forms"])
        self.X_dim = int(metadata["X_dim"])
        self.brand_forms = list(metadata["brand_forms"])
        self.intercept_names = list(metadata["intercept_names"])
        self.numeric_names = list(metadata["numeric_names"])
        self.random_feature_indices = list(metadata["random_feature_indices"])
        self.n_draws = int(metadata["n_draws"])
        self.k_rand = (self.n_brand_forms - 1) + len(self.random_feature_indices)
        self.n_fixed_features = len(
            [i for i in range(1, self.X_dim) if i - 1 not in self.random_feature_indices]
        )

        draws = make_std_normal_draws(
            self.n_households, self.n_draws, self.k_rand, metadata["random_seed"]
        )
        if fingerprint_tensors([draws]) != metadata["draws_fingerprint"]:
            raise RuntimeError(
                "The regenerated random draws differ from those of step 1 (e.g. after a "
                "PyTorch upgrade). Rerun step1_prepare_data.py --force with a new "
                "results root."
            )
        self.std_normal_draws = draws.to(self.device)
        self._batches = {}
        print(
            f"Data: {metadata['n_observations']} purchases, {self.n_households} households, "
            f"{self.n_brand_forms} brand-forms, X_dim={self.X_dim}, k_rand={self.k_rand}"
        )

    def batches(self, n_batches: int) -> list:
        if n_batches not in self._batches:
            self._batches[n_batches] = split_data_by_household(
                self.data_dict, self.std_normal_draws, n_batches=n_batches
            )
        return self._batches[n_batches]

    def release_batches(self, *n_batches) -> None:
        for n in (n_batches or list(self._batches)):
            self._batches.pop(n, None)
        gc.collect()
        empty_cache(self.device)

    def model_kwargs(self, family: str, include_uniquenesses_rand: bool, emulator=None) -> dict:
        kwargs = {
            "n_brand_forms": self.n_brand_forms,
            "n_total_households": self.n_households,
            "X_dim": self.X_dim,
            "random_feature_indices": list(self.random_feature_indices),
            "include_uniquenesses_rand": include_uniquenesses_rand,
            "n_draws": self.n_draws,
            "device": self.device,
            "model_family": family,
            "anchored_rand": True,
        }
        if emulator is not None:
            kwargs["emulator"] = emulator
        return kwargs


# =============================================================================
# Emulators
# =============================================================================

def load_emulators(device, names=None, checkpoints=None) -> dict:
    """Load emulator checkpoints (wrapped in DataParallel when several GPUs exist)."""
    from training import load_checkpoint

    checkpoints = EMULATOR_CHECKPOINTS if checkpoints is None else checkpoints
    names = emulators_to_fit() if names is None else names
    emulators = {}
    for name in names:
        emulator, _, _ = load_checkpoint(checkpoints[name])
        emulators[name] = emulator.to(device).eval()
        print(f"Loaded emulator '{name}' from {checkpoints[name]}")

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count > 1:
        for name in emulators:
            emulators[name] = torch.nn.DataParallel(
                emulators[name], device_ids=list(range(device_count))
            )
        print(f"Emulators wrapped in DataParallel over {device_count} GPUs")
    return emulators


# =============================================================================
# Checkpoints and coefficient tables
# =============================================================================

def load_summary_if_exists(path: str):
    """Saved model summary at `path`, or None if it has not been written yet."""
    if not os.path.exists(path):
        return None
    summary = load_model_summary(path)
    version = summary.get("config", {}).get("param_version")
    if version != PARAM_VERSION:
        raise RuntimeError(
            f"{path} was written with parameter layout version {version!r} "
            f"(current: {PARAM_VERSION}). Move old results away before resuming."
        )
    return summary


def require_summary(path: str, hint: str):
    summary = load_summary_if_exists(path)
    if summary is None:
        raise FileNotFoundError(f"{path} not found. {hint}")
    return summary


def summary_metrics(summary: dict) -> dict:
    return {
        "loglike": float(summary["log_likelihood"]),
        "aic": float(summary["aic"]),
        "bic": float(summary["bic"]),
        "caic": float(summary["caic"]),
        "n_params": int(summary["n_params"]),
        "converged": bool(summary["converged"]),
        "elapsed_time": float(summary.get("elapsed_time", float("nan"))),
    }


def _write_csv_atomic(df, path: str) -> None:
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path)
    os.replace(tmp_path, path)


def _record_coef_timing(timing_path: str, entry: dict) -> None:
    timing = load_model_summary(timing_path) if os.path.exists(timing_path) else {"times": []}
    identity = {k: v for k, v in entry.items() if k != "elapsed_time"}
    timing["times"] = [
        t for t in timing["times"]
        if {k: v for k, v in t.items() if k != "elapsed_time"} != identity
    ] + [entry]
    timing["total_time"] = float(sum(t["elapsed_time"] for t in timing["times"]))
    save_model_summary(timing, timing_path)


def ensure_coef_table(
    ctx: AnalysisContext,
    summary_path: str,
    table_path: str,
    model_kwargs: dict,
    timing_path: str = None,
    timing_entry: dict = None,
) -> bool:
    """
    Make sure a model has its coefficient table (CSV) and its joint covariance
    matrix (stored in the JSON summary). Returns True if anything was computed.

    The covariance is computed once; if only the CSV is missing it is rebuilt
    from the saved covariance.
    """
    summary = require_summary(summary_path, "Fit the model first.")
    has_covariance = summary.get("covariance") is not None
    if has_covariance and os.path.exists(table_path):
        return False

    start = time.time()
    table, covariance = create_coefficient_table_from_summary(
        summary_path=summary_path,
        batch_data_dicts=ctx.batches(N_BATCHES_COEF_TABLES),
        model_kwargs=model_kwargs,
        intercept_names=ctx.intercept_names,
        numeric_names=ctx.numeric_names,
        return_covariance=True,
    )
    elapsed = time.time() - start

    if not has_covariance:
        add_covariance_to_summary(summary, covariance)
        summary["covariance_elapsed_time"] = float(elapsed)
        save_model_summary(summary, summary_path)
    _write_csv_atomic(table, table_path)
    print(f"  Coefficient table saved: {table_path} ({elapsed:.1f}s)")

    if timing_path is not None and timing_entry is not None:
        _record_coef_timing(timing_path, {**timing_entry, "elapsed_time": float(elapsed)})

    del table, covariance
    gc.collect()
    empty_cache(ctx.device)
    return True


def mark_as_final(summary: dict, model_name: str, **config_updates) -> dict:
    """Copy of `summary` renamed to `model_name`, with extra config entries."""
    final = copy.deepcopy(summary)
    final["model_name"] = model_name
    final["config"].update(config_updates)
    return final


# =============================================================================
# Plots
# =============================================================================

def plot_factor_comparison(rows: list, family: str, path: str) -> None:
    """-2 loglik / AIC / BIC / CAIC against the number of random-effects factors."""
    x = [row["n_factors"] for row in rows]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, [-2.0 * row["loglike"] for row in rows], "o-", label="-2 * log-likelihood")
    ax.plot(x, [row["aic"] for row in rows], "s-", label="AIC")
    ax.plot(x, [row["bic"] for row in rows], "^-", label="BIC")
    ax.plot(x, [row["caic"] for row in rows], "d-", label="CAIC")
    ax.legend()
    ax.set_xlabel("Random Effects: # Factors")
    ax.set_title(f"{FAMILIES[family]['label']} Model Comparison ({len(rows)} models)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_emulator_bic(bics: dict, tf_list: list, path: str, baselines: dict = None,
                      title: str = "Emulator Model Comparison by BIC") -> None:
    """BIC against the number of time factors, one line per emulator."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, values in bics.items():
        points = [(tf, v) for tf, v in zip(tf_list, values) if v is not None]
        if points:
            ax.plot(*zip(*points), marker="o", linestyle="-",
                    color=EMULATOR_COLORS.get(name, "tab:gray"), label=f"{name} (dense)")
    styles = [("black", "--"), ("gray", ":")]
    for (label, value), (color, style) in zip((baselines or {}).items(), styles):
        ax.axhline(y=value, color=color, linestyle=style, linewidth=1.5, label=label)
    ax.legend(loc="best")
    ax.set_xlabel("Time-Specific Errors: # Factors")
    ax.set_ylabel("BIC")
    ax.set_title(title)
    ax.set_xticks(tf_list)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)