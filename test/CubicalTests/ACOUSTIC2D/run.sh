#!/bin/bash
#SBATCH --job-name=acoustic_test
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --account=ARLAP14877100
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --chdir=/p/home/grauta/git/CombinatorialSpaces.jl/test/CubicalTests/ACOUSTIC2D/
#SBATCH --output=slurm_logs/acoustic_test_%j.out
#SBATCH --error=slurm_err_logs/acoustic_test_%j.err

module unload PrgEnv-cray
module load PrgEnv-cray
module load cray-hdf5-parallel

# ── User-defined variables ────────────────────────────────
cd $SIM_DIR

WX=1
WY=1
OX=1
OY=1
N_TOTAL=$((WX*WY + OX*OY))

RUNTAG=w${WX}x${WY}_o${OX}x${OY}

PROJECT_DIR=/p/home/grauta/git/CombinatorialSpaces.jl
SIM_DIR=${PROJECT_DIR}/test/CubicalTests/ACOUSTIC2D
LOG_DIR=${SIM_DIR}/${RUNTAG}/${SLURM_JOB_ID}
SCRIPT=${SIM_DIR}/main.jl

mkdir -p ${LOG_DIR}

mpiexecjl -n $N_TOTAL julia --project=${PROJECT_DIR} $SCRIPT $WX $WY $OX $OY $LOG_DIR