#!/bin/bash

# Suppress all Python warnings globally
export PYTHONWARNINGS="ignore"

# This script runs several train.py experiments for the "meld" task.

# Define common parameters here:

WP=2
BSZ=1
ACC_STEP=8
LR=2e-4
PTMLR=1e-5
DPT=0.3

# Each command is executed sequentially.
'''
echo "Running: MELD baseline model without query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "meld" -uq "False" -mt "baseline"

echo "Running: MELD baseline model with query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "meld" -uq "True" -mt "baseline"

echo "Running: MELD CRF model without query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "meld" -uq "False" -mt "CRF"
'''
echo "Running: MELD CRF model with query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "meld" -uq "True" -mt "CRF"

echo "Running: IEMOCAP baseline model without query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "iemocap" -uq "False" -mt "baseline"

echo "Running: IEMOCAP baseline model with query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "iemocap" -uq "True" -mt "baseline"

echo "Running: IEMOCAP CRF model without query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "iemocap" -uq "False" -mt "CRF"

echo "Running: IEMOCAP CRF model with query..."
python train.py -tr -wp $WP -bsz $BSZ -acc_step $ACC_STEP -lr $LR -ptmlr $PTMLR -dpt $DPT -tsk "iemocap" -uq "True" -mt "CRF"