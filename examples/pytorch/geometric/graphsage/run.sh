# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -ex

ALGO=${1:-supervised}
ADL_UPLOADER=${2:-no}
## DEVICE support: ["cpu", "gpu"]
DEVICE=${3:-cpu}
STORAGE_TYPE=${4:-memory}
USE_HADOOP=${5:-no}

DIR_NAME=$(dirname "$0")

GRAPH=/tmp/reddit
#rm -fr $GRAPH

#python -m deepgnn.graph_engine.data.citation --data_dir $GRAPH

MODEL_DIR=/tmp/model_fix
rm -rf $MODEL_DIR

if [[ "${DEVICE}" == "gpu" ]]
then
    PLATFORM_DEVICE=--gpu
fi

### ===== training =======
python ${DIR_NAME}/main.py  \
--data_dir $GRAPH --mode train --seed 123 \
--backend snark --graph_type remote --converter skip \
--batch_size 512 --learning_rate 0.005 --num_epochs 5 \
--node_type 0 --max_id -1 \
--model_dir $MODEL_DIR --metric_dir $MODEL_DIR --save_path $MODEL_DIR \
--feature_idx 1 --feature_dim 300 --label_idx 0 --label_dim 50 --algo $ALGO \
--log_by_steps 5 --use_per_step_metrics $PLATFORM_DEVICE --storage_type $STORAGE_TYPE \
--server_idx 0 --client_rank 0 --num_ge 1 --data_parallel_num 2


rm -rf $MODEL_DIR
