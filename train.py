
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

parser = argparse.ArgumentParser(description='AlphaZero Training')
parser.add_argument('--framework', choices=('tf', 'torch'), required=True)
parser.add_argument('--game', type=str, required=True)
parser.add_argument('-his', '--history', type=int, default=3)
parser.add_argument('-l',   '--layers', type=int, default=20)
parser.add_argument('-f',   '--filters', type=int, default=256)
parser.add_argument('-hf',  '--head-filters', type=int, default=1)
parser.add_argument('-lr',  '--learning-rate', type=float, default=1e-3)
parser.add_argument('-wd',  '--weight-decay', type=float, default=1e-4)
parser.add_argument('-sim', '--sim-count', type=int, default=1600)
parser.add_argument('-epi', '--ep-count', type=int, default=25000)
parser.add_argument('-epo', '--epochs', type=int, default=1)
parser.add_argument('-b',   '--batch-size', type=int, default=4096)
parser.add_argument('-s',   '--step-size', type=int, default=128)
parser.add_argument('-buf', '--buffer-size', type=int, default=20)
parser.add_argument('-cp',  '--c-puct', type=float, default=2.5)
parser.add_argument('-a',   '--alpha', type=float, default=0.03)
parser.add_argument('-eps', '--epsilon', type=float, default=0.25)
parser.add_argument('-cut', '--cutoff', type=int, default=30)
parser.add_argument('-td',  '--td-epsilon', type=float, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start-iter', type=int, default=0)
parser.add_argument('--n-iter', type=int, default=100)
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--log-dir', type=str, default='./checkpoints')
parser.add_argument('--log-match', action='store_true', default=False)
args = parser.parse_args()

if args.framework == 'tf':
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    device = None
else:
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
exec('from models.model_{} import Agent'.format(args.framework))
exec('from cy_{} import self_play, get_game_details, ReplayBuffer'.format(args.game))

def main():
    np.random.seed(args.seed)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    board_height, board_width, ac_dim, state_depth, const_depth = get_game_details()
    agent = Agent(
        args.log_dir, args.layers, args.filters, args.head_filters,
        args.weight_decay, board_height, board_width, ac_dim,
        args.history * state_depth + const_depth,
        args.learning_rate, args.td_epsilon, device)
    rb = ReplayBuffer(args.log_dir, args.history, args.buffer_size, args.td_epsilon)
    if args.start_iter > 0:
        agent.load(args.start_iter-1)
        if args.framework == 'torch':
            for g in agent.opt.param_groups:
                g['lr'] = args.learning_rate
        rb.load(args.start_iter-1)
        loss_df = pd.read_csv('{}/loss{}.csv'.format(
            args.log_dir, '_td' if args.td_epsilon else ''))
    else:
        loss_df = pd.DataFrame({'loss' : []})
    run_training_loop(agent, rb, loss_df)

def run_training_loop(agent, rb, loss_df):
    start_time = time.time()
    for i in range(args.start_iter, args.start_iter + args.n_iter):
        print('\n====================Iter {}===================='.format(i))
        print('Self-playing {} episodes...'.format(args.ep_count))
        if args.framework == 'torch':
            agent.nnet.eval()
        print('Black: {}, White: {}, Draw: {}'.format(
            *self_play(
                a_ep_count=args.ep_count, a_sim_count=args.sim_count, a_tau=1,
                eval_func_p1=agent.forward, eval_func_p2=agent.forward,
                rb=rb, a_c_puct=args.c_puct, a_alpha=args.alpha, a_epsilon=args.epsilon,
                a_cutoff=args.cutoff, a_td=args.td_epsilon, a_save=1, a_log=0, file=None)))
        print('Time elapsed: {:.3f}s'.format(time.time()-start_time))
    
        print('\nUpdating parameters...')
        if args.framework == 'torch':
            agent.nnet.train()
        loss = np.mean([agent.update(*sample) \
            for sample in rb.sample(args.epochs, args.batch_size, args.step_size)])
        loss_df.loc[i] = loss
        loss_df.to_csv('{}/loss{}.csv'.format(
            args.log_dir, '_td' if args.td_epsilon else ''), index=False)
        print('Loss: {:.3f}'.format(loss))
        print('Time elapsed: {:.3f}s'.format(time.time()-start_time))

        rb.mark_end(i)
        agent.save(i)
        rb.save(i)

        if args.log_match:
            print('\nLogging sample match...')
            if args.framework == 'torch':
                agent.nnet.eval()
            with open('{}/{:03d}/sample_match{}.txt'.format(
                    args.log_dir, i, '_td' if args.td_epsilon else ''), 'w') as file:
                self_play(
                    a_ep_count=1, a_sim_count=args.sim_count, a_tau=0,
                    eval_func_p1=agent.forward, eval_func_p2=agent.forward,
                    rb=rb, a_c_puct=args.c_puct, a_alpha=args.alpha, a_epsilon=args.epsilon,
                    a_cutoff=args.cutoff, a_td=args.td_epsilon, a_save=0, a_log=1, file=file)
            print('Logging complete.')
            print('Time elapsed: {:.3f}s'.format(time.time()-start_time))
  
    print('\nLoop completed with {} iterations. Total time elapsed: {}s.' \
        .format(args.n_iter, int(time.time()-start_time+1)))

if __name__ == '__main__':
    main()
