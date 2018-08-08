#!/usr/bin/env bash

set -ex

# run labelling UI
./label_ui.py \
    --image-dir sample_data/training/ \
    --label-db sample_data/labels.db \
    --width 768 --height 1024

# materialise label database into bitmaps
./materialise_label_db.py \
    --label-db sample_data/labels.db \
    --directory sample_data/labels/ \
    --width 768 --height 1024

# generate some samples of the data
./data.py \
    --image-dir sample_data/training/ \
    --label-dir sample_data/labels/ \
    --width 768 --height 1024

# train for a bit...
./train.py \
    --run r12 \
    --steps 300 \
    --train-steps 50 \
    --train-image-dir sample_data/training/ \
    --test-image-dir sample_data/test/ \
    --label-dir sample_data/labels/ \
    --width 768 --height 1024

# run inference against unlabelled data
./predict.py \
    --run r12 \
    --image-dir sample_data/unlabelled \
    --output-label-db predictions.db \
    --export-pngs predictions
