import argparse
from utils import init_dir, set_seed
from meta_trainer import MetaTrainer
import os
from subgraph import gen_subgraph_datasets
import time


def run(args):
    if args.kge in ['TransE', 'DistMult', 'TDistMult']:
        args.ent_dim = args.dim
        args.rel_dim = args.dim
        args.time_dim = args.dim

    elif args.kge == 'RotatE':
        args.ent_dim = args.dim * 2
        args.rel_dim = args.dim
        args.time_dim = args.dim
    elif args.kge in ['ComplEx', 'TComplEx']:
        args.ent_dim = args.dim * 2
        args.rel_dim = args.dim * 2
        args.time_dim = args.dim
    elif args.kge == 'TeRo':
        args.ent_dim = args.dim * 2
        args.rel_dim = args.dim * 2
        args.time_dim = args.dim * 2

    args.db_path = args.data_path[:-4] + '_subgraph'

    if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args)

    init_dir(args)
    # ----------------------------------------
    # Full
    args.exp_name = args.task_name

    trainer = MetaTrainer(args)
    trainer.train()

    del trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data/icews14/day/icews14_ext.pkl') #icews05-15 icews14 day month
    parser.add_argument('--state_dir', default='./state')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--tb_log_dir', default='./tb_log')
    parser.add_argument('--seed', default=1314, type=int)

    parser.add_argument('--task_name', default='icews14_ext')
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--num_exp', default=1, type=int)

    parser.add_argument('--train_bs', default=64, type=int)
    parser.add_argument('--eval_bs', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_step', default=100000, type=int) #100000
    parser.add_argument('--log_per_step', default=10, type=int) #10
    parser.add_argument('--check_per_step', default=30, type=int) #30
    parser.add_argument('--early_stop_patience', default=20, type=int) #20
    parser.add_argument('--num_sample_cand', default=5, type=int)

    parser.add_argument('--dim', default=32, type=int)
    parser.add_argument('--ent_dim', default=None, type=int)
    parser.add_argument('--rel_dim', default=None, type=int)
    parser.add_argument('--time_dim', default=None, type=int)
    parser.add_argument('--num_layers', default=4, type=int)

    # parser.add_argument('--num_rel_bases', default=4, type=int)
    parser.add_argument('--gcn_drop', default=0.2, type=float)

    parser.add_argument('--kge', default='TeRo', type=str, choices=['TransE', 'DistMult', 'ComplEx', 'TDistMult', 'TComplEx', 'RotatE', 'TeRo'])
    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--cl_temp', default=0.05, type=float)

    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=0.001, type=float)

    parser.add_argument('--cpu_num', default=2, type=float)
    parser.add_argument('--gpu', default='cuda:0', type=str)

    # subgraph
    parser.add_argument('--db_path', default=None)
    parser.add_argument('--num_train_subgraph', default=10000)
    parser.add_argument('--num_sample_for_estimate_size', default=10)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)



    args = parser.parse_args()

    set_seed(args.seed)

    run(args)

