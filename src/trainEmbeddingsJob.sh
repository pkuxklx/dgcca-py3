#!/bin/sh
#$ -cwd
#$ -l mem_free=32g
#$ -l h_rt=24:00:00
#$ -l h_vmem=32g
#$ -l num_proc=8
#$ -N train.multi.emb
#$ -S /bin/bash

IN_DIR="/export/projects/abenton/multiviewTweetRepresentations/data/dgcca_10-7-2016/"
OUT_DIR="/export/projects/abenton/multiviewTweetRepresentations/experiments/dgcca/"

# Twitter user raw data
IN_PATH="/export/projects/abenton/multiviewTweetRepresentations/data/hashtag_prediction/user_6views_tfidf_pcaEmbeddings_userTweets_network.tsv.gz"
TRAIN_PATH="${IN_DIR}/user_6views.full.train.npz"
TUNE_PATH="${IN_DIR}/user_6views.full.tune.npz"

# Some reasonable defaults
#lr=0.005 # Too big a step size for many of the models
lr=0.0001
activation="sigmoid"
bsize=4000
#bsize=1000
#numEpochs=50
numEpochs=200

# Training parameters
architecture=${1} # Set architecture of each view
k=${2}
rcov=${3}
l1=${4}
l2=${5}
vnameStr=${6}
numEpochs=${7}

# drop spaces
archStr=${architecture// /}

# Where to write output
BASE="embedding_avg_dgcca_k=${k}_rcov=${rcov}_arch=${archStr}_l1=${l1}_l2=${l2}"
EMBEDDINGS_PATH="${OUT_DIR}/user_embeddings/${BASE}.embedding.npz"
MODEL_PATH="${OUT_DIR}/models/${BASE}.model.npz"
LOG_PATH="${OUT_DIR}/logs/${BASE}.log.txt"
HISTORY_PATH="${OUT_DIR}/logs/${BASE}.history.npz"

echo "python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch \"${architecture}\" --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} ${rcov} ${rcov} --learningRate ${lr} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --epochs ${numEpochs} --lcurvelog ${HISTORY_PATH} --warmstart | tee -a ${LOG_PATH}"
python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch "${architecture}" --k ${k} --rcov ${rcov} ${rcov} ${rcov} ${rcov} ${rcov} ${rcov} --learningRate ${lr} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --epochs ${numEpochs} --lcurvelog ${HISTORY_PATH} --warmstart | tee -a ${LOG_PATH}
