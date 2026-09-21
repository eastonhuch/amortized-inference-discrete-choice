# amortized-inference-discrete-choice

Code implementing the methods of "Amortized Inference for Correlated Discrete Choice Models via Equivariant Neural Networks" by Easton Huch and Michael Keane.

## Setup

```bash
pip install -r requirements.txt
```

The figures are rendered with LaTeX (`text.usetex` in `config_plt.py`), so a LaTeX installation is required to produce them.

As mentioned in the paper, we fit the analysis emulators and analyze the application data using GPUs, so we recommend GPUs for the case study.
For the simulation study, we trained the emulators using Apple Metal Performance Shaders (MPS) and ran the estimation on CPU.

## Repository layout

| Path | Contents |
| --- | --- |
| `emulator.py` | Equivariant neural network emulator |
| `training.py` | Data simulation, emulator training, and checkpoint loading |
| `utils.py`, `config_plt.py` | Shared helpers and plotting configuration |
| `simulation.py`, `run_simulation.py` | Simulation study: emulator vs. GHK estimation |
| `fixed_case_accuracy.py` | Emulator accuracy and break-even simulation sizes at fixed parameter values |
| `analysis.py`, `pipeline_common.py`, `step*.py`, `dense_time_summaries.py` | Case-study estimation pipeline |
| `nielsen_shared.py`, `nielsen-*.ipynb` | Case-study data preparation and summaries |
| `fit-*.ipynb`, `cross-k.ipynb`, `visualize-training.ipynb` | Emulator training and training diagnostics |
| `pregenerate_buffers.py` | Pre-generates training data for the analysis emulators |
| `model-checkpoints/` | Trained emulators used in the paper |
| `simulation-results/` | Simulation-study output (logs, CSV results, LaTeX tables) |
| `figures/` | Figures produced by the code |

## Simulation study

```bash
bash run_simulation.sh
```

This script

1. trains the simulation emulators (`fit-simulation-emulators.ipynb`, `fit-gumbel-simulation-emulator.ipynb`, and the multi-K emulator in `cross-k.ipynb`);
2. runs the simulation study (`run_simulation.py`) for each setting, writing logs, CSV results, and LaTeX tables to `simulation-results/`;
3. plots the training diagnostics (`visualize-training.ipynb`);
4. runs the fixed-case accuracy comparison (`fixed_case_accuracy.py`).

Training writes per-step logs of each emulator fit to `fitting-results/`.
The trained emulators used in the paper are included in `model-checkpoints/`.

## Case study

To preserve confidentiality, this repository does not contain the data from the case study.
However, the data is available via participating academic institutions through the Kilts Center for Marketing at the University of Chicago Booth School of Business.

1. Set the `BASE_DIR` variable in `nielsen_shared.py` to the directory where you have saved the NielsenIQ data. The data should be organized into `panel` and `scanner` subfolders, each containing folders named 2021, 2022, 2023, and 2024 with the extracted laundry detergent data.
2. Run `nielsen-market-share.ipynb`.
3. Run `nielsen-panel.ipynb`.
4. Execute `run_analysis.sh`. This script trains the analysis emulators (pre-generating their training data with `pregenerate_buffers.py`) and then runs the estimation pipeline:
   - `step1_prepare_data.py`: filters households and prepares the choice data;
   - `step2_fit_exact_models.py`: exact mixed probit and mixed logit models;
   - `step3_fit_emulator_models.py`: emulator-based models with time-specific factor covariances;
   - `step3b_fit_dense_time_models.py`: emulator-based models with a dense time-specific covariance;
   - `step4_summaries.py`: tables, likelihood-ratio tests, and cross-evaluations.
5. Run `nielsen-summaries.ipynb` for the data summary table, the model comparison table, and the diversion-ratio figures.

The pipeline writes its results to `analysis-results/`, or to the directory given by the environment variable `ANALYSIS_RESULTS_ROOT`.
This directory contains a processed copy of the panel data (`prepared/data.pt`), so keep it in the same protected location as the raw data.
Each step skips models whose results are already on disk, so an interrupted run can be resumed by rerunning the step.
Settings are in the configuration section of `pipeline_common.py`.

The trained analysis emulators are included in `model-checkpoints/`, so the emulator training at the start of step 4 can be skipped by running only the `step*.py` commands at the end of `run_analysis.sh`.

## Citation

If you use this code, please cite the paper ([arXiv:2603.24705](https://arxiv.org/abs/2603.24705)):

```bibtex
@misc{huch2026amortized,
  title         = {Amortized Inference for Correlated Discrete Choice Models via Equivariant Neural Networks},
  author        = {Huch, Easton and Keane, Michael},
  year          = {2026},
  eprint        = {2603.24705},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2603.24705}
}
```

## License

The source code is released under the MIT license (see `LICENSE`).
The license does not apply to results derived from NielsenIQ data (for example, the case-study tables and figures and the outputs in `nielsen-*.ipynb`), which remain subject to the terms of use of the NielsenIQ Datasets at the Kilts Center for Marketing.

## Data disclaimer

All reported empirical results are from our own analyses using data from Nielsen Consumer LLC and marketing databases provided through the NielsenIQ Datasets at the Kilts Center for Marketing Data Center at the University of Chicago Booth School of Business.
The conclusions drawn from the NielsenIQ data are our own and do not reflect the views of NielsenIQ; NielsenIQ is not responsible for, had no role in, and was not involved in analyzing or preparing the results reported herein.
