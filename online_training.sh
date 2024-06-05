rdir='/home/Sergey.Frolov/work/model_error/work/online_prototype/'
outputDir=${rdir}'/ifs'
sd="2021-01-02T00"
ed="2021-01-12T00"
inDir='/scratch2/BMC/gsienkf/Laura.Slivinski/model_error_corr_work/data/subsample/'
nnBefore="${rdir}/checks/pc_conv2d_${vname}_4_1_4096_3_0.25_32_mse_0.0001_1.0_364_363_0.7"
GPUPYTHON='/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python'

#generate npy training dataset from netcdf files
# this s done once for all variables 
python code/online_training.py -mode prep -s $sd -e $ed -i $inDir -o $outputDir -n 40

#for each t,q,u,v train a network
# this can be done in parallel by submiting each varn job on a separate node
vname='t'
python code/online_training.py -mode train -s $sd -e $ed -i $inDir -o $outputDir -nnBefore $nnBefore -vname=$vname


