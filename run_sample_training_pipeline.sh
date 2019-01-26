#!/usr/bin/env bash

# this script represents the end to end pipeline from
# labelling -> training -> various testing / predict methods

# it runs NOWHERE near long enough during training to get a decent
# result and is included as a smoke test

set -x

rm -rf sample_data/labels/ ckpts/r12 tb/r12 sample_predictions.db

set -e

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

# generate some 256x236 sample patches of the data.
./data.py \
    --image-dir sample_data/training/ \
    --label-dir sample_data/labels/ \
    --rotate --distort \
    --patch-width-height 256

# train for a bit using 256 square patches for training and
# full resolution for test.
./train.py \
    --run r12 \
    --steps 2 \
    --train-steps 2 \
    --batch-size 4 \
    --train-image-dir sample_data/training/ \
    --test-image-dir sample_data/test/ \
    --label-dir sample_data/labels/ \
    --pos-weight 5 \
    --patch-width-height 256 \
    --width 768 --height 1024

# run inference against unlabelled data
./predict.py \
    --run r12 \
    --image-dir sample_data/unlabelled \
    --output-label-db sample_predictions.db \
    --export-pngs predictions

# check loss statistics against training data
./test.py \
    --run r12 \
    --image-dir sample_data/training/ \
    --label-db sample_data/labels.db

# check loss statistics against labelled test data
./test.py \
    --run r12 \
    --image-dir sample_data/test/ \
    --label-db sample_data/labels.db
