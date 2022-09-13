#!/bin/bash

python3 train.py \
        --model-name segmentation_gta5_vgg16 \
        --segmentation-training-loaders "gta5_train" \
        --segmentation-resize-height 576 \
        --segmentation-resize-width 1024 \
        --segmentation-training-batch-size 12