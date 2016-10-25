#!/bin/sh
#$ -cwd
#$ -l mem_free=4g
#$ -l h_rt=24:00:00
#$ -l h_vmem=4g
#$ -l num_proc=8
#$ -N train.multi_emb.example
#$ -S /bin/bash

# Example script to learn DGCCA embeddings on an SGE grid.
# Adrian Benton 10/18/2016

OUT_DIR="../test/"
mkdir ${OUT_DIR}

# Some examples 
RSC_DIR="../resources/"
IN_PATH="${RSC_DIR}/sample_wgcca_input.tsv.gz"
TRAIN_PATH="${RSC_DIR}/sample_wgcca_input.train.npz"
TUNE_PATH="${RSC_DIR}/sample_wgcca_input.tune.npz"

# Some reasonable defaults
lr=0.01
bsize=1000
rcov=0.000001
adam_opt="{\"type\":\"adam\",\"params\":{\"adam_b1\":0.1,\"adam_b2\":0.001,\"learningRate\":${lr}}}"
sgd_momentum_opt="{\"type\":\"sgd_momentum\",\"params\":{\"momentum\":0.99,\"decay\":1.0,\"learningRate\":${lr}}}"
sgd_opt="{\"type\":\"sgd\",\"params\":{\"decay\":1.0,\"learningRate\":${lr}}}"
l1=0.0001
l2=0.01
numEpochs=50
valFreq=5

architecture="[[1000,250,500],[1000,250,500],[1000,250,500],[1000,250,500]]"
k=100
activation="relu"
vnameStr="view1 view2 view3 view4"
vweightStr="1.0 1.0 1.0 0.1"

# Where to write output
BASE="embedding_avg_dgcca_act=${activation}_k=${k}_rcov=${rcov}_arch=${archStr}_l1=${l1}_l2=${l2}"
EMBEDDINGS_PATH="${OUT_DIR}/${BASE}.embedding.npz"
MODEL_PATH="${OUT_DIR}/${BASE}.model.npz"
LOG_PATH="${OUT_DIR}/${BASE}.log.txt"
HISTORY_PATH="${OUT_DIR}/${BASE}.history.npz"

# Trains model with three different optimizers, each warm-started with best best weights
# from previous run
echo "python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch \"${architecture}\" --truncparam 1000 --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt \"${adam_opt}\" --epochs ${numEpochs} --valfreq ${valFreq} --lcurvelog ${HISTORY_PATH} | tee ${LOG_PATH}"
python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch "${architecture}" --truncparam 500 --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt "${adam_opt}" --epochs ${numEpochs} --valfreq ${valFreq} --lcurvelog ${HISTORY_PATH} | tee ${LOG_PATH}
echo "python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch \"${architecture}\" --truncparam 1000 --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt \"${sgd_momentum_opt}\" --epochs ${numEpochs} --valfreq ${valFreq} --lcurvelog ${HISTORY_PATH} --warmstart | tee -a ${LOG_PATH}"
python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch "${architecture}" --truncparam 500 --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt "${sgd_momentum_opt}" --epochs ${numEpochs} --valfreq ${valFreq} --lcurvelog ${HISTORY_PATH} --warmstart | tee -a ${LOG_PATH}
echo "python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch \"${architecture}\" --truncparam 1000 --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt \"${sgd_opt}\" --epochs ${numEpochs} --valfreq ${valFreq} --lcurvelog ${HISTORY_PATH} --warmstart | tee -a ${LOG_PATH}"
python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch "${architecture}" --truncparam 500 --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt "${sgd_opt}" --epochs ${numEpochs} --valfreq ${valFreq} --lcurvelog ${HISTORY_PATH} --warmstart | tee -a ${LOG_PATH}
