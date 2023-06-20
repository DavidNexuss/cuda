#!/bin/sh

### Directivas para el gestor de colas
#SBATCH --job-name=deviceQuery
#SBATCH -D .
#SBATCH --output=submit-deviceQuery.o%j
#SBATCH --error=submit-deviceQuery.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

export PATH=/Soft/cuda/11.2.1/bin:$PATH
RUN=./run.sh

$RUN scripts/test1.c scripts/runFrame.c
$RUN scripts/testSphere.c scripts/runFrame.c
$RUN scripts/test3.c scripts/runFrame.c
$RUN scripts/test4.c scripts/runFrame.c

echo "Run movie generation test (may take several minutes)"
read trash
$RUN scripts/test1.c scripts/runFrames.c
