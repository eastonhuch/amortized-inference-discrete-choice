#!/bin/bash
# Fits the analysis emulators (GPU recommended), then runs the case-study pipeline.

echo 'Pre-generating probit data for analysis'
python pregenerate_buffers.py --output-dir 'probit-analysis-data-checkpoints' --num-factors 8

echo 'Fitting probit emulator for analysis'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output=fit-probit-analysis-emulator.ipynb fit-probit-analysis-emulator.ipynb

cp model-checkpoints/checkpoint_probit_2_3_4_5_6_7_8_9_10_11_12.pt model-checkpoints/probit12.pt

echo 'Pre-generating Gumbel data for analysis'
python pregenerate_buffers.py --gumbel --output-dir 'gumbel-analysis-data-checkpoints' --num-factors 8

echo 'Fitting Gumbel emulator for analysis'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output=fit-gumbel-analysis-emulator.ipynb fit-gumbel-analysis-emulator.ipynb

cp model-checkpoints/checkpoint_gumbel_2_3_4_5_6_7_8_9_10_11_12.pt model-checkpoints/gumbel12.pt

echo 'Fitting large Gumbel emulator for analysis'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output=fit-gumbel-large-analysis-emulator.ipynb fit-gumbel-large-analysis-emulator.ipynb

cp model-checkpoints/checkpoint_gumbel-large_2_3_4_5_6_7_8_9_10_11_12.pt model-checkpoints/gumbel-large12.pt

# Dense emulators: trained on full-rank (dense) covariance matrices (--num-factors -1)
echo 'Pre-generating dense probit data for analysis'
python pregenerate_buffers.py --output-dir 'probit-dense-analysis-data-checkpoints' --num-factors -1 --k-max 12

echo 'Fitting dense probit emulator for analysis'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output=fit-probit-dense-analysis-emulator.ipynb fit-probit-dense-analysis-emulator.ipynb

cp model-checkpoints/checkpoint_probit_dense_2_3_4_5_6_7_8_9_10_11_12.pt model-checkpoints/probit_dense12.pt

echo 'Pre-generating dense Gumbel data for analysis'
python pregenerate_buffers.py --gumbel --output-dir 'gumbel-dense-analysis-data-checkpoints' --num-factors -1 --k-max 12

echo 'Fitting dense Gumbel emulator for analysis'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output=fit-gumbel-dense-analysis-emulator.ipynb fit-gumbel-dense-analysis-emulator.ipynb

cp model-checkpoints/checkpoint_gumbel_dense_2_3_4_5_6_7_8_9_10_11_12.pt model-checkpoints/gumbel_dense12.pt

echo 'Fitting large dense Gumbel emulator for analysis'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output=fit-gumbel-large-dense-analysis-emulator.ipynb fit-gumbel-large-dense-analysis-emulator.ipynb

cp model-checkpoints/checkpoint_gumbel-large_dense_2_3_4_5_6_7_8_9_10_11_12.pt model-checkpoints/gumbel-large_dense12.pt

python step1_prepare_data.py          # filter households, build data_dict, save a data cache
python step2_fit_exact_models.py      # mixed probit/logit factor sequences, BIC choice, dense models
python step3_fit_emulator_models.py   # emulator models over time factors, with the gumbel refinement
python step3b_fit_dense_time_models.py  # dense time covariance with the dense emulators
python step4_summaries.py             # BIC plot, results tables, LR tests, cross-evaluation
