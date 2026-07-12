#!/bin/bash
#SBATCH --job-name=acoustic3d
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --account=ARLAP14877100
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --chdir=/p/home/grauta/git/CombinatorialSpaces.jl/test/CubicalTests/ACOUSTIC3D/
#SBATCH --output=slurm_logs/acoustic_test_%j.out
#SBATCH --error=slurm_err_logs/acoustic_test_%j.err

module unload PrgEnv-cray
module load PrgEnv-cray
module load cray-hdf5-parallel

# ── User-defined variables ────────────────────────────────
cd $SIM_DIR

WX=1
WY=1
WZ=1
OX=1
OY=1
OZ=1
N_TOTAL=$((WX*WY*WZ + OX*OY*OZ))

RUNTAG=w${WX}x${WY}x${WZ}_o${OX}x${OY}x${OZ}

PROJECT_DIR=/p/home/grauta/git/CombinatorialSpaces.jl
SIM_DIR=${PROJECT_DIR}/test/CubicalTests/ACOUSTIC3D
LOG_DIR=${SIM_DIR}/${RUNTAG}/${SLURM_JOB_ID}
SCRIPT=${SIM_DIR}/main.jl

mkdir -p ${LOG_DIR}

# cp config.toml ${LOG_DIR}/config.toml

mpiexecjl -n $N_TOTAL julia --project=${PROJECT_DIR} $SCRIPT $WX $WY $WZ $OX $OY $OZ $LOG_DIR