#!/bin/bash
#SBATCH -J deqn_sweep_3D_09_omp_simd_stripmine 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --partition=haswell64

module load scorep
module load hdeem

BENCHMARK="../../../src/3D/09_omp_simd_stripmine/build/deqn3d.x"
echo "Slurm job id: ${SLURM_JOB_ID}"
cat /proc/cpuinfo
clearHdeem
cd $(dirname $BENCHMARK)
startHdeem
srun ./$(basename $BENCHMARK)
stopHdeem
printHdeem -o deqn_sweep_3D_09_omp_simd_stripmine_${SLURM_JOB_ID}_hdeem.csv
clearHdeem
