
from cy_othello import ReplayBuffer, self_play
import numpy as np
import pandas as pd
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

if __name__ == '__main__':
    def greedy(state):
        prior = np.full(state.shape[0] * 65, 1.0 / 65, dtype=np.float32)
        flatstate = state.reshape((state.shape[0], state.shape[1], -1)).astype(np.float32)
        value = (np.sum(flatstate[:, -3, :] - flatstate[:, -2, :], axis=1) * \
                (flatstate[:, -1, 0] * 2 - 1) / 64).astype(np.float32)
        return prior, value

    start_time = time.time()
    with open('./demo.txt', 'w') as file:
        result = self_play(a_ep_count=1, a_sim_count=800, a_tau=1,
            eval_func_p1=greedy, eval_func_p2=greedy, rb=ReplayBuffer(history=3),
            a_c_puct=4.0, a_alpha=0.8, a_epsilon=0.25, a_cutoff=10, a_td=0, a_log=1, file=file)
    elapsed = time.time() - start_time
    print('Black: {}, White: {}, Draw: {}'.format(*result))
    print('Time elapsed: {:.3f}s'.format(elapsed))
