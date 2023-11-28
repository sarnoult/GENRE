#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DATASET=$1
NAME=$2

fairseq-train $DATASET/bin/ \
    --save-dir models/$NAME \
    --tensorboard-logdir tensorboard_logs/$NAME \
    --restore-file models/fairseq_multilingual_entity_disambiguation/model.pt \
    --arch bart_large  \
    --task translation  \
    --criterion label_smoothed_cross_entropy  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens 128  \
    --update-freq 1  \
    --max-update 20  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01  \
    --optimizer adam  \
    --adam-betas "(0.9, 0.999)"  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-05  \
    --total-num-update 20  \
    --warmup-updates 5  \
    --ddp-backend no_c10d  \
    --num-workers 10  \
    --reset-meters  \
    --reset-optimizer \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --patience 2 \
    --encoder-normalize-before \
    --decoder-normalize-before
