#!/bin/sh
#SBATCH --nodes 1
#SBATCH -A gsienkf
#SBATCH -p fgewf
#SBATCH --qos windfall
#SBATCH -t 8:0:0

#salloc -t 8:0:0 -A gsienkf -p fgewf --qos=windfall -N 1

$RUNDIR='/scratch2/BMC/gsienkf/Laura.Slivinski/modelerror_2/'
cd $RUNDIR
echo $PWD


cd code/model_error_correction_with_ai/
echo $PWD
source ./setenv.sh
cd -

#$GPUPYTHON code/sequential_training.py
$GPUPYTHON batch_training_parallel_GPU.py
 

