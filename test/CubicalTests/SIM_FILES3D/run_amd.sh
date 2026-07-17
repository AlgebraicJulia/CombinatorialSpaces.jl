#!/bin/bash
#SBATCH --job-name=acoustic_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi300a:4
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --account=ARLAP14877100
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --chdir=/p/home/grauta/git/CombinatorialSpaces.jl/test/CubicalTests/SIM_FILES3D/
#SBATCH --output=slurm_logs/test_%j.out
#SBATCH --error=slurm_err_logs/test_%j.err

module unload PrgEnv-cray
module load PrgEnv-amd
module load cray-hdf5-parallel

export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD=/opt/cray/pe/mpich/9.1.0/ofi/amd/7.0/lib/libmpi_gtl_hsa.so

# ── User-defined variables ────────────────────────────────
cd $SIM_DIR

WX=$1
WY=$2
WZ=$3
OX=1
OY=1
OZ=1
N_TOTAL=$((WX*WY*WZ + OX*OY*OZ))

RUNTAG=w${WX}x${WY}x${WZ}_o${OX}x${OY}x${OZ}

PROJECT_DIR=/p/home/grauta/git/CombinatorialSpaces.jl
SIM_DIR=${PROJECT_DIR}/test/CubicalTests/SIM_FILES3D
SCRIPT=${SIM_DIR}/main.jl

# 3 is sim name and 4 is a switch for compute device
mpiexecjl -n $N_TOTAL julia --project=${PROJECT_DIR} $SCRIPT $WX $WY $WZ $OX $OY $OZ $4 $5