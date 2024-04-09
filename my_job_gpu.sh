#!/bin/sh
#SBATCH --nodes 1
#SBATCH -A gsienkf
#SBATCH -p fge
#SBATCH --qos gpuwf
#SBATCH -t 8:0:0

#salloc -t 8:0:0 -A gsienkf -p fge --qos=gpuwf -N 1

$RUNDIR='set-to-something-meaningfull'
cd $RUNDIR
echo $PWD


cd code
echo $PWD
source ./setenv.sh
cd -

#$GPUPYTHON code/sequential_training.py
$GPUPYTHON batch_training_parallel_GPU.py
 

