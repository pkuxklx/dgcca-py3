#!/bin/sh
#$ -cwd
#$ -l mem_free=16g
#$ -l h_rt=24:00:00
#$ -l h_vmem=16g
#$ -l num_proc=8
#$ -N train.multi.emb
#$ -S /bin/bash

IN_DIR="/WHERE/I/STORE/NUMPY/DATA"
OUT_DIR="/WHERE/I/STORE/EXPERIMENTS"

# Raw data
IN_PATH="/PATH/TO/INPUT/VIEWS.tsv.gz"
TRAIN_PATH="${IN_DIR}/user_6views.full.train.npz"
TUNE_PATH="${IN_DIR}/user_6views.full.tune.npz"

# Some reasonable defaults
#lr=0.005 # Too big a step size for many of the models
lr=0.0001
activation="relu"
bsize=1024 # minibatch size
numEpochs=200
truncParam=1000 # How many left singular vectors to keep in our data matrices
valfreq=1 # Number of epochs between checking reconstruction error

# Optimizers
adam_opt="{\"type\":\"adam\",\"params\":{\"adam_b1\":0.1,\"adam_b2\":0.001,\"learningRate\":${lr}}}"
sgd_momentum_opt="{\"type\":\"sgd_momentum\",\"params\":{\"momentum\":0.99,\"decay\":1.0,\"learningRate\":${lr}}}"
sgd_opt="{\"type\":\"sgd\",\"params\":{\"decay\":1.0,\"learningRate\":${lr}}}"

# Training parameters
architecture=${1} # Set architecture of each view
k=${2}
rcovStr=${3}
l1=${4}
l2=${5}
vnameStr=${6}
numEpochs=${7}
vweights=${8} # How much to weight each view

# drop spaces
archStr=${architecture// /}
vweightStr=${vweights// /,}
rcovStrCat=${rcovStr// /,}

# Where to write output
BASE="embedding_dgcca_wts=${vweights}_k=${k}_rcov=${rcovStrCat}_arch=${archStr}_l1=${l1}_l2=${l2}"

mkdir "${OUT_DIR}/embeddings"
mkdir "${OUT_DIR}/models"
mkdir "${OUT_DIR}/logs"

EMBEDDINGS_PATH="${OUT_DIR}/embeddings/${BASE}.embedding.npz"
MODEL_PATH="${OUT_DIR}/models/${BASE}.model.npz"
LOG_PATH="${OUT_DIR}/logs/${BASE}.log.txt"
HISTORY_PATH="${OUT_DIR}/logs/${BASE}.history.npz"

echo "python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch \"${architecture}\" --truncparam ${truncParam} --k ${k} --rcov ${rcovStrCat} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt \"${adam_opt}\" --epochs ${numEpochs} --valfreq ${valfreq} --lcurvelog ${HISTORY_PATH} | tee ${LOG_PATH}"
python dgcca_train_harness.py --input ${IN_PATH} --preptrain ${TRAIN_PATH} --preptune ${TUNE_PATH} --output ${EMBEDDINGS_PATH} --model ${MODEL_PATH} --arch \"${architecture}\" --truncparam ${truncParam} --k ${k} --rcov ${rcovStrCat} --batchSize ${bsize} --l1 ${l1} --l2 ${l2} --vnames ${vnameStr} --activation ${activation} --weights ${vweightStr} --opt \"${adam_opt}\" --epochs ${numEpochs} --valfreq ${valfreq} --lcurvelog ${HISTORY_PATH} | tee ${LOG_PATH}
