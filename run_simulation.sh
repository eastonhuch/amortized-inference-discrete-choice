#!/bin/bash

echo 'Fitting emulators for simulation study'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 fit-simulation-emulators.ipynb

echo 'Running simulation study'

echo '3'
python run_simulation.py \
    --checkpoint "./model-checkpoints/checkpoint_emulator_3.pt" \
    --k 3 \
    --num-reps 100 \
    --sample-sizes 1000 10000 100000 \
    > simulation-results/simulation-3.out

echo '5'
python run_simulation.py \
    --checkpoint "./model-checkpoints/checkpoint_emulator_5.pt" \
    --k 5 \
    --num-reps 100 \
    --sample-sizes 1000 10000 100000 \
    > simulation-results/simulation-5.out

echo '10'
python run_simulation.py \
    --checkpoint "./model-checkpoints/checkpoint_emulator_10.pt" \
    --k 10 \
    --num-reps 100 \
    --sample-sizes 1000 10000 100000 \
    > simulation-results/simulation-10.out

echo '3 factor'
python run_simulation.py \
    --checkpoint "./model-checkpoints/checkpoint_emulator_3.pt" \
    --k 3 \
    --factor \
    --num-reps 100 \
    --sample-sizes 1000 10000 100000 \
    > simulation-results/simulation-3-factor.out

echo '5 factor'
python run_simulation.py \
    --checkpoint "./model-checkpoints/checkpoint_emulator_5.pt" \
    --k 5 \
    --num-reps 100 \
    --factor \
    --sample-sizes 1000 10000 100000 \
    > simulation-results/simulation-5-factor.out

echo '10 factor'
python run_simulation.py \
    --checkpoint "./model-checkpoints/checkpoint_emulator_10.pt" \
    --k 10 \
    --factor \
    --num-reps 100 \
    --sample-sizes 1000 10000 100000 \
    > simulation-results/simulation-10-factor.out

echo 'Fitting emulators for simulation study'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 fit-gumbel-simulation-emulator.ipynb

echo '5 gumbel'
python run_simulation.py \
    --checkpoint "./model-checkpoints/checkpoint_emulator_gumbel_5.pt" \
    --k 5 \
    --errors gumbel \
    --sim-configs 5:500 5:2000 20:500 20:2000 \
    --sim-memory-gb 8 \
    --num-reps 100 \
    --sample-sizes 1000 10000 100000 \
    > simulation-results/simulation-5-gumbel.out

echo 'Fitting multi-k emulator'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 cross-k.ipynb

echo 'Visualizing emulator fits'
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 visualize-training.ipynb

python fixed_case_accuracy.py