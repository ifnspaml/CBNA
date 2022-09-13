#!/bin/bash

python3 adapt.py \
      --sys-best-effort-determinism \
      --model-name "cbna_adaptation" \
      --model-load segmentation_gta5_vgg16/checkpoints/epoch_20 \
      --model-disable-lr-loading \
      --adaptation-batch-size 1 \
      --adaptation-loaders "cityscapes_sequence" \
      --adaptation-resize-height 512 \
      --adaptation-resize-width 1024 \
      --adaptation-bn-momentum 0.2 \
      --adaptation-batches 500 \
      --model-cbna-bn-inference