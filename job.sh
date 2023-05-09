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
./runBoada.sh scripts/test1.c scripts/runFrame.c
./runBoada.sh scripts/test2.c scripts/runFrame.c
./runBoada.sh scripts/test3.c scripts/runFrame.c
