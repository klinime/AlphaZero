
import argparse
import numpy as np
import time
import sys

np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='Model Competition')
parser.add_argument('--framework', choices=('tf', 'torch'), required=True)
parser.add_argument('--game', type=str, required=True)
parser.add_argument('--load', action='store_true', default=False)
parser.add_argument('--iter-nums', type=int, nargs='+', required=True)
parser.add_argument('--log-dir', type=str, default='./checkpoints')
parser.add_argument('-his', '--history', type=int, default=3)
parser.add_argument('-l',   '--layers', type=int, default=20)
parser.add_argument('-f',   '--filters', type=int, default=256)
parser.add_argument('-hf',  '--head-filters', type=int, default=1)
parser.add_argument('-sim', '--sim-count', type=int, default=1600)
parser.add_argument('-epi', '--ep-count', type=int, default=25000)
parser.add_argument('-cp',  '--c-puct', type=float, default=2.5)
parser.add_argument('-a',   '--alpha', type=float, default=0.03)
parser.add_argument('-eps', '--epsilon', type=float, default=0.25)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--output', type=str, default='./scores.npy')
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
    board_height, board_width, ac_dim, state_depth, const_depth = get_game_details()
    assert len(args.iter_nums) > 1
    iter_nums = args.iter_nums if len(args.iter_nums) > 3 \
        else np.arange(*args.iter_nums)
    agents = [Agent(
        args.log_dir, args.layers, args.filters, args.head_filters,
        0, board_height, board_width, ac_dim,
        args.history * state_depth + const_depth, 0, 0, device).load(iter_nums[0])]
    rb = ReplayBuffer(args.log_dir, args.history, 1, 0)

    start_time = time.time()
    scores = np.load(args.output) if args.load \
        else np.zeros((iter_nums.size, iter_nums.size, 3), dtype=np.int)
    for i in range(1, iter_nums.size):
        if len(agents) == i:
            agents.append(Agent(
                args.log_dir, args.layers, args.filters, args.head_filters, 
                0, board_height, board_width, ac_dim,
                args.history * state_depth + const_depth,
                0, 0, device).load(iter_nums[i]))
        for j in range(i):
            if np.any(scores[i, j]):
                continue

            print('P{} vs P{}: {} episodes...'.format(
                iter_nums[i], iter_nums[j], args.ep_count//2))
            result = self_play(
                a_ep_count=args.ep_count//2, a_sim_count=args.sim_count, a_tau=0,
                eval_func_p1=agents[i].forward, eval_func_p2=agents[j].forward, rb=rb,
                a_c_puct=args.c_puct, a_alpha=args.alpha, a_epsilon=args.epsilon)
            print('Black: {}, White: {}, Draw: {}'.format(*result))
            scores[i, j, 0] += result[0]
            scores[i, j, 1] += result[1]
            scores[i, j, 2] += result[2]
            print('Time elapsed: {:.3f}s\n'.format(time.time() - start_time))

            print('P{} vs P{}: {} episodes...'.format(
                iter_nums[j], iter_nums[i], args.ep_count//2))
            result = self_play(
                a_ep_count=args.ep_count//2, a_sim_count=args.sim_count, a_tau=0,
                eval_func_p1=agents[j].forward, eval_func_p2=agents[i].forward, rb=rb,
                a_c_puct=args.c_puct, a_alpha=args.alpha, a_epsilon=args.epsilon)
            print('Black: {}, White: {}, Draw: {}'.format(*result))
            scores[i, j, 0] += result[1]
            scores[i, j, 1] += result[0]
            scores[i, j, 2] += result[2]
            print('Time elapsed: {:.3f}s\n'.format(time.time() - start_time))
            
            np.save(args.output, scores)
        print(scores[:i+1, :i+1, 0] + scores[:i+1, :i+1, 1].T)

if __name__ == '__main__':
    main()
