# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

DIR_NAME=$(dirname "$0")

MODEL_DIR=/tmp/model_fix
rm -rf $MODEL_DIR

### ===== training =======
python ${DIR_NAME}/main.py  \
--data_dir /tmp/reddit --mode train --seed 123 \
--backend snark --graph_type local --converter skip \
--batch_size 512 --learning_rate 0.005 --num_epochs 10 \
--node_type 0 --max_id 152410 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
--feature_idx 1 --feature_dim 300 --label_idx 0 --label_dim 50 --algo supervised \
--log_by_steps 1 --use_per_step_metrics --fanouts 25 25
