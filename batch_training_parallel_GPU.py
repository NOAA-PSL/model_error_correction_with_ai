#!/scratch1/NCEPDEV/global/Tse-chun.Chen/anaconda3/envs/ltn/bin/python
import torch
import training as t
import importlib
import time
importlib.reload(t)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ptmp=[device, 't', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 363,  364, 0.7] #(rank,vars_out, testset, kernel_sizes, channels, n_conv, p, bs, loss_name, lr, wd, end_of_training_day, training_validation_length_days, tv_ratio)
ptmp=[device, 'q', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 363,  364, 0.7] 
ptmp=[device, 'u', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 363,  364, 0.7] 
ptmp=[device, 'v', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 363,  364, 0.7] 
ptmp=[device, 'ps', 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., 363,  364, 0.7] 

st = time.time()
t._train_(*ptmp)
et = time.time()
print('Execution time:', et-st, 'seconds')



