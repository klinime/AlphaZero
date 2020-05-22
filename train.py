
import argparse
import numpy as np
from pathlib import Path
import time

parser = argparse.ArgumentParser(description='AlphaZero Training')
parser.add_argument('--framework', choices=('tf', 'torch'), required=True)
parser.add_argument('--game', type=str, required=True)
parser.add_argument('-his', '--history', type=int, default=3)
parser.add_argument('-l',   '--n_layers', type=int, default=20)
parser.add_argument('-f',   '--filters', type=int, default=256)
parser.add_argument('-hf',  '--head_filters', type=int, default=1)
parser.add_argument('-lr',  '--learning_rate', type=float, default=1e-3)
parser.add_argument('-wd',  '--weight_decay', type=float, default=1e-4)
parser.add_argument('-sim', '--sim_count', type=int, default=1600)
parser.add_argument('-epi', '--ep_count', type=int, default=25000)
parser.add_argument('-epo', '--epochs', type=int, default=1)
parser.add_argument('-b',   '--batch_size', type=int, default=4096)
parser.add_argument('-s',   '--step_size', type=int, default=128)
parser.add_argument('-buf', '--buffer_size', type=int, default=20)
parser.add_argument('-td',  '--td_epsilon', type=float, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start_iter', type=int, default=0)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--log_dir', type=str, default='./checkpoints')
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
path = args.log_dir
Path(path).mkdir(parents=True, exist_ok=True)

def main(params):
	np.random.seed(params['seed'])

	history = params['history']
	n_layers = params['n_layers']
	filters = params['filters']
	head_filters = params['head_filters']
	c = params['weight_decay']
	lr = params['learning_rate']
	buffer_size = params['buffer_size']
	td = params['td_epsilon']

	start_iter = params['start_iter']
	n_iter = params['n_iter']
	ep_count = params['ep_count']
	sim_count = params['sim_count']
	epochs = params['epochs']
	batch_size = params['batch_size']
	step_size = params['step_size']

	board_height, board_width, ac_dim, state_depth = get_game_details()
	agent = Agent(
		path, n_layers, filters, head_filters, c, board_height,
		board_width, ac_dim, state_depth * history, lr, td, device)
	rb = ReplayBuffer(path, history, buffer_size, td)
	if start_iter > 0:
		agent.load(start_iter-1)
		rb.load(start_iter-1)

	run_training_loop(
		agent, rb, start_iter, n_iter, ep_count,
		sim_count, td, epochs, batch_size, step_size)

def run_training_loop(
		agent, rb, start_iter, n_iter, ep_count,
		sim_count, td, epochs, batch_size, step_size):

	start_time = time.time()
	for i in range(start_iter, start_iter + n_iter):
		print('\n====================Iter {}===================='.format(i))
		print('Self-playing {} episodes...'.format(ep_count))
		if args.framework == 'torch':
			agent.nnet.eval()
		print('Black: {}, White: {}, Draw: {}'.format(
			*self_play(
				a_ep_count=ep_count, a_sim_count=sim_count, a_tau=1,
				eval_func_p1=agent.forward, eval_func_p2=agent.forward,
				rb=rb, a_td=td, a_save=1, a_log=0, file=None)))
		print('Time elapsed: {:.3f}s'.format(time.time()-start_time))
	
		print('\nUpdating parameters...')
		if args.framework == 'torch':
			agent.nnet.train()
		loss = np.mean([agent.update(*sample) \
			for sample in rb.sample(epochs, batch_size, step_size)])
		with open('{}/loss{}.txt'.format(path, '_td' if td else ''), 'a') as file:
			file.write('\n{}'.format(loss))
		print('Loss: {:.3f}'.format(loss))
		print('Time elapsed: {:.3f}s'.format(time.time()-start_time))

		rb.mark_end(i)
		agent.save(i)
		rb.save(i)

		print('\nLogging sample match...')
		if args.framework == 'torch':
			agent.nnet.eval()
		with open('{}/{:03d}/sample_match{}.txt' \
				.format(path, i, '_td' if td else ''), 'w') as file:
			self_play(
				a_ep_count=1, a_sim_count=sim_count, a_tau=0,
				eval_func_p1=agent.forward, eval_func_p2=agent.forward,
				rb=rb, a_td=0, a_save=0, a_log=1, file=file)
		print('Logging complete.')
		print('Time elapsed: {:.3f}s'.format(time.time()-start_time))
  
	print('\nLoop completed with {} iterations. Total time elapsed: {}s.' \
		.format(n_iter, int(time.time()-start_time+1)))

if __name__ == '__main__':
	main(vars(args))
