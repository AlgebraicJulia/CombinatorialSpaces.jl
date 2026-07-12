#!/bin/bash
#SBATCH --job-name=LMNSH
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --account=ARLAP14877100
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:mi300a:4
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/p/home/grauta/git/CombinatorialSpaces.jl/test/CubicalTests/SIM_FILES2D/
#SBATCH --output=amdlogs/slurm_logs/test_%j.out
#SBATCH --error=amdlogs/slurm_err_logs/test_%j.err

EXPECTED=5

if [ "$#" -ne "$EXPECTED" ]; then
    echo "Error: expected $EXPECTED arguments, got $#." >&2
    echo "Usage: $(basename "$0") <simulation name> <wx> <wy> <ox> <oy>" >&2
    exit 1
fi

module swap PrgEnv-cray PrgEnv-amd
module load cray-hdf5-parallel

# ── User-defined variables ────────────────────────────────
cd $SIM_DIR

WX=$2
WY=$3
OX=$4
OY=$5
N_TOTAL=$((WX*WY + OX*OY))

# RUNTAG=w${WX}x${WY}_o${OX}x${OY}

PROJECT_DIR=/p/home/grauta/git/CombinatorialSpaces.jl
SIM_DIR=${PROJECT_DIR}/test/CubicalTests/SIM_FILES2D
# LOG_DIR=${SIM_DIR}/${RUNTAG}/${SLURM_JOB_ID}
LOG_DIR=${SIM_DIR}/amdlogs/${SLURM_JOB_ID}
SCRIPT=${SIM_DIR}/main.jl

mkdir -p ${LOG_DIR}

mpiexecjl -n $N_TOTAL julia --project=${PROJECT_DIR} $SCRIPT $WX $WY $OX $OY $LOG_DIR $1