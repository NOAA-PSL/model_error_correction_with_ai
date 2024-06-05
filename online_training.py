import argparse, os
import preprocess as pp
from datetime import datetime

import torch
import training as t
#import importlib
#importlib.reload(t)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Pre-process netcdf files")

    parser.add_argument("-s", dest="start", help="start date yyyy-mm-dd-Thh")
    parser.add_argument("-e", dest="end", help="end date yyyy-mm-dd-Thh")
    parser.add_argument("-i", dest="inDir", help="input directory")
    parser.add_argument("-o", dest="outputDir", help="output directory")
    parser.add_argument("-n", dest="nJobs", default=1, type=int, help="number of parallel jobs")
    parser.add_argument("-nnBefore", dest="nnBefore", default=None, help="fname for the pre-trained network")
    parser.add_argument("-mode", dest="mode", choices=['prep', 'train'], help="mode: prep or train ")
    parser.add_argument("-vname", dest="vname", choices=['t','q','u','v'], help="variable name for training: t,q,u,v")


    args = parser.parse_args()
    print(args)

    #compute number of days in training
    date1 = datetime.strptime(args.start,'%Y-%m-%dT%H')
    date2 = datetime.strptime(args.end,'%Y-%m-%dT%H')

    ndays = date2-date1
    ndays = ndays.days

    # execute
    if args.mode=='prep':
        # pre-process data
        pp.preprocess(args)
    elif args.mode=='train':
        #train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ptmp=[device, args.vname, 4, '1', '4096', 3, 0.25, 32, 'mse', 0.0001, 1., ndays,  ndays, 0.7, args.outputDir]

        #reset pre-trained network if available
        if (os.path.exists(args.nnBefore)) :
            fn_this = t.create_checkpoint_filename(ptmp[1:-1])
            t.reset_network(args.nnBefore, fn_this)
        t._train_(*ptmp)
        print(f"done training {fn_this}")

