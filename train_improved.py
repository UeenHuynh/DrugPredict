from argparse import Namespace
from logging import Logger
import numpy as np
import os
import pandas as pd
from dmfpga.train import fold_train
from dmfpga.tool import set_log, mkdir
from dmfpga.tool import set_train_argument
import joblib
import torch, gc

def training_improved(args, log):
    info = log.info
    mkdir(args.save_path)
    joblib.dump(args, os.path.join(args.save_path, 'train_args.pkl'))    
    
    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path
    
    score = []
    val_score = []
    train_score = []
    
    for num_fold in range(args.num_folds):
        info(f'Seed {args.seed}')
        args.seed = seed_first + num_fold
        args.save_path = os.path.join(save_path, f'Seed_{args.seed}')
        mkdir(args.save_path)
        
        fold_score, fold_val_score, fold_train_score = fold_train(args, log)
        
        score.append(fold_score)
        val_score.append(fold_val_score)
        train_score.append(fold_train_score)
    
    score = np.array(score)
    val_score = np.array(val_score)
    train_score = np.array(train_score)
    
    info(f'Running {args.num_folds} folds in total.')
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):
           info(f'test {args.metric} = {np.nanmean(fold_score):.6f}')

    score_ave = np.nanmean(score, axis=0)
    score_std = np.nanstd(score)
    val_score_ave = np.nanmean(val_score, axis=0)
    train_score_ave = np.nanmean(train_score, axis=0)

    info(f'Average train {args.metric} = {train_score_ave[4]:.6f}'
         f'  acc = {train_score_ave[0]:.6f}'
         f'  precision = {train_score_ave[1]:.6f}'
         f'  recall = {train_score_ave[2]:.6f}')

    info(f'Average val {args.metric} = {val_score_ave[4]:.6f}'
         f'  acc = {val_score_ave[0]:.6f}'
         f'  precision = {val_score_ave[1]:.6f}'
         f'  recall = {val_score_ave[2]:.6f}'
         f'  specificity = {val_score_ave[3]:.6f}')
         
    info(f'test {args.metric} = {score_ave[4]:.6f} ± {np.nanstd(score[:, 4]):.6f}'
         f'  acc = {score_ave[0]:.6f} ± {np.nanstd(score[:, 0]):.6f}'
         f'  precision = {score_ave[1]:.6f} ± {np.nanstd(score[:, 1]):.6f}'
         f'  recall = {score_ave[2]:.6f} ± {np.nanstd(score[:, 2]):.6f}'
         f'  specificity = {score_ave[3]:.6f} ± {np.nanstd(score[:, 3]):.6f}')
    
    return score_ave, score_std

if __name__ == '__main__':
    args = set_train_argument()
    
    # Cải thiện hyperparameters - cân bằng hơn
    args.num_epochs = 30  # Giảm epochs
    args.batch_size = 32  # Batch size vừa phải
    args.nhid = 64        # Hidden units vừa phải
    args.dropout = 0.2    # Dropout thấp hơn
    args.lr = 0.001       # Learning rate
    args.patience = 15    # Tăng patience
    
    mkdir(args.save_path)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    log = set_log('train_improved', args.log_path)
    
    training_improved(args, log)