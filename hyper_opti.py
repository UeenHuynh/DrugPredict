# from argparse import ArgumentParser, Namespace
# from hyperopt import fmin, tpe, hp
# import numpy as np
# import os
# from copy import deepcopy
# # from dmfpga.tool import set_hyper_argument, set_log
# from dmfpga.tool import set_train_argument as set_hyper_argument, get_task_name, set_log
# from train import training

# space = {
#          'fp_2_dim':hp.quniform('fp_2_dim', low=300, high=600, q=50),
#          'nhid':hp.quniform('nhid', low=40, high=80, q=5),
#          'nheads':hp.quniform('nheads', low=2, high=8, q=1),
#          'gat_scale':hp.quniform('gat_scale', low=0.2, high=0.8, q=0.1),
#          'dropout':hp.quniform('dropout', low=0.0, high=0.6, q=0.05),
#          'dropout_gat':hp.quniform('dropout_gat', low=0.0, high=0.6, q=0.05)
# }


# def fn(space):
#     search_no = args.search_now
#     log_name = 'train'+str(search_no)
#     log = set_log(log_name,args.log_path)
#     result_path = os.path.join(args.log_path, 'hyper_para_result.txt')
    
#     list = ['fp_2_dim','nhid','nheads']
#     for one in list:
#         space[one] = int(space[one])
#     hyperp = deepcopy(args)
#     name_list = []
#     change_args = []
#     for key,value in space.items():
#         name_list.append(str(key))
#         name_list.append('-')
#         name_list.append((str(value))[:5])
#         name_list.append('-')
#         setattr(hyperp,key,value)
#     dir_name = "".join(name_list)
#     dir_name = dir_name[:-1]
#     hyperp.save_path = os.path.join(hyperp.save_path, dir_name)
    
#     ave,std = training(hyperp,log)
    
#     with open(result_path,'a') as file:
#         file.write(str(space)+'\n')
#         file.write('Result '+str(hyperp.metric)+' : '+str(ave)+' +/- '+str(std)+'\n')
    
#     if ave is None:
#         if hyperp.dataset_type == 'classification':
#             ave = 0
#         else:
#             raise ValueError('Result of model is error.')
    
#     args.search_now += 1
    
#     if hyperp.dataset_type == 'classification':
#         return -ave
#     else:
#         return ave

# def hyper_searching(args):
#     result_path = os.path.join(args.log_path, 'hyper_para_result.txt')
    
#     result = fmin(fn,space,tpe.suggest,args.search_num)
    
#     with open(result_path,'a') as file:
#         file.write('Best Hyperparameters : \n')
#         file.write(str(result)+'\n')
        

# if __name__ == '__main__':
#     args = set_hyper_argument()
#     hyper_searching(args)
    
# hyper_opti.py

import os
from copy import deepcopy
from argparse import ArgumentParser

import numpy as np
from hyperopt import fmin, tpe, hp

from dmfpga.tool import set_train_argument, get_task_name, set_log
from train import training


def parse_args():
    """
    1) Parse your standard training args via set_train_argument()
    2) Add HyperOpt-specific flags (max_evals, search_now)
    """
    base_args = set_train_argument()

    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        '--max_evals',
        type=int,
        default=50,
        help='Number of hyperparameter evaluations'
    )
    parser.add_argument(
        '--search_now',
        type=int,
        default=0,
        help='Internal counter for current evaluation'
    )
    extra, _ = parser.parse_known_args()
    
    # merge into base_args
    base_args.max_evals    = extra.max_evals
    base_args.search_num   = extra.max_evals
    base_args.search_now   = extra.search_now
    return base_args


# ─── Step 1: load args ─────────────────────────────────────────────────────────
args = parse_args()


# ─── Step 2: define search space ───────────────────────────────────────────────
space = {
    'fp_2_dim':  hp.quniform('fp_2_dim',  low=300, high=600, q=50),
    'nhid':      hp.quniform('nhid',      low=40,  high=80,  q=5),
    'nheads':    hp.quniform('nheads',    low=2,   high=8,   q=1),
    'gat_scale': hp.quniform('gat_scale', low=0.2, high=0.8, q=0.1),
    'dropout':   hp.quniform('dropout',   low=0.0, high=0.6, q=0.05),
    'dropout_gat': hp.quniform('dropout_gat', low=0.0, high=0.6, q=0.05),
}


# ─── Step 3: objective function ────────────────────────────────────────────────
def objective(hparams):
    # 3.1 convert to int where needed
    for key in ('fp_2_dim', 'nhid', 'nheads'):
        hparams[key] = int(hparams[key])

    # 3.2 prepare a fresh copy of args and inject hyperparameters
    hyperp = deepcopy(args)
    for k, v in hparams.items():
        setattr(hyperp, k, v)

    # 3.3 build a unique save_path based on hyperparameters
    segments = [f"{k}-{v}" for k, v in hparams.items()]
    dir_name = "_".join(segments)
    hyperp.save_path = os.path.join(hyperp.save_path, dir_name)
    os.makedirs(hyperp.save_path, exist_ok=True)

    # 3.4 setup a logger for this eval
    base_task = get_task_name(args.data_path).split('.')[0]
    log_name  = f"{base_task}_{args.search_now}"
    logger    = set_log(log_name, args.log_path)
    logger.info(f"Starting eval #{args.search_now} with {hparams}")

    # 3.5 run training & capture metrics
    # ave, std = training(hyperp, logger)
    ave, std = training(hyperp, logger)

    # --- extract the single scalar for the metric ---
    # assume args.metric is e.g. "auc" → index 4, or "acc" → index 0, etc.
    metric = hyperp.metric.lower()
    idx_map = {
        'acc':       0,
        'accuracy':  0,
        'precision': 1,
        'presion':   1,
        'recall':    2,
        'spe':       3,
        'auc':       4
    }
    idx = idx_map.get(metric, 0)
    mean_val = float(ave[idx]) if isinstance(ave, (list, tuple, np.ndarray)) else float(ave)
    std_val  = float(std[idx]) if isinstance(std, (list, tuple, np.ndarray)) else float(std)
 
    # 3.6 record to a central results file
    # result_file = os.path.join(args.log_path, 'hyper_para_result.txt')
    # always write the hyperparameter results under save_path (which is a directory)
    os.makedirs(args.save_path, exist_ok=True)
    result_file = os.path.join(args.save_path, 'hyper_para_result.txt')
    with open(result_file, 'a') as f:
        f.write(f"{hparams}\n")
        f.write(f"Result {hyperp.metric}: {mean_val:.4f} ± {std_val:.4f}\n\n")

    # 3.7 increment counter and return objective
    args.search_now += 1
    if hyperp.dataset_type == 'classification':
        return -mean_val  # maximize accuracy
    return -mean_val     # minimize error


# ─── Step 4: run the search ────────────────────────────────────────────────────
if __name__ == '__main__':
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals
    )

    # write out the best hyperparameters
    # result_file = os.path.join(args.log_path, 'hyper_para_result.txt')
    # with open(result_file, 'a') as f:
    # after search finishes, append best params into the same file under save_path
    result_file = os.path.join(args.save_path, 'hyper_para_result.txt')
    with open(result_file, 'a') as f:    
        f.write("=== Best Hyperparameters ===\n")
        f.write(f"{best}\n")
