
# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
import pandas as pd
from scipy.special import softmax
from libc.stdint cimport uint8_t, uint16_t
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr, weak_ptr, make_shared
from libcpp.string cimport string
from cython.operator cimport dereference as deref

cdef extern from "../games/othello.hpp" namespace "othello" nogil:
    cdef cppclass Game:
        Game() except +
        Game(uint8_t turn) except +
        uint8_t get_turn()
        int get_board_dim()
        int get_board_size()
        int get_ac_dim()
        int get_state_depth()
        vector[uint8_t] get_state()
        vector[uint16_t] get_action()
        unique_ptr take_action(uint16_t)
        int terminal()
        int evaluate()
        string state_str()
        vector[string] action_str()

cdef extern from "mcts.hpp" namespace "mcts" nogil:
    cdef cppclass Node:
        weak_ptr[Node] m_parent
        vector[shared_ptr[Node]] m_children
        shared_ptr[Game] m_game
        int   m_visit
        float m_value
        vector[float] m_prior
        Node() except +
        Node(shared_ptr[Node], unique_ptr) except +
    
    vector[vector[uint8_t]] get_eval_states(vector[shared_ptr[Node]]&, int, int, int)
    void select(vector[shared_ptr[Node]]&, vector[shared_ptr[Node]]&,
                int, float, float, float, int)
    vector[float] expand(vector[shared_ptr[Node]]&)
    void backprop(vector[shared_ptr[Node]]&, vector[shared_ptr[Node]]&,
                  vector[float]&, vector[float]&, int)

##############################################################################
############################### Replay Buffer ################################
##############################################################################

cdef class ReplayBuffer:
    cdef:
        str path
        np.ndarray s, pi, z
        int length
        list idx
        int capacity
        float td
    cdef readonly int history

    def __init__(self, str path="./", int history=3, int capacity=0, float td=0):
        self.path = path
        self.s = None
        self.pi = None
        self.z = None
        self.length = 0
        self.idx = []
        self.history = history
        self.capacity = capacity
        self.td = td
    
    cpdef void add_history(self, np.ndarray[np.uint8_t, ndim=3] state,
                                 np.ndarray[np.float32_t, ndim=1] policy, float value):
        # appends state and policy to s and pi, optionally value if td > 0
        cdef:
            np.ndarray[np.uint8_t, ndim=4] s = np.expand_dims(state, axis=0)
            np.ndarray[np.float32_t, ndim=2] pi = np.expand_dims(policy, axis=0)
        
        self.length += 1
        if (self.s is None):
            self.s = s
            self.pi = pi
            if (self.td):
                self.z = np.full((1, 1), value, dtype=np.float32)
        else:
            self.s = np.append(self.s, s, axis=0)
            self.pi = np.append(self.pi, pi, axis=0)
            if (self.td):
                self.z = np.append(self.z, np.full(
                    (1, 1), value, dtype=np.float32), axis=0)
    
    cpdef void update_result(self, int winner):
        # updates the z for winner at the end of a game
        cdef:
            np.ndarray[np.int8_t, ndim=2] z
            np.ndarray[np.float32_t, ndim=2] v
        z = np.ones((self.length, 1), dtype=np.int8)
        z[(self.history & 1) :: 2] = -1
        z = winner * z
        if (self.td):
            v = self.z[self.z.shape[0] - self.length : self.z.shape[0] - 1] - \
                self.z[self.z.shape[0] - self.length + 1:]
            self.z[self.z.shape[0] - self.length : self.z.shape[0] - 1] = \
                (1 - self.td) * z[:self.length - 1] + self.td * v
            self.z[self.z.shape[0] - 1, 0] = z[self.length - 1, 0]
        else:
            self.z = z if self.z is None else np.concatenate((self.z, z))
        self.length = 0
    
    cpdef void extend(self, ReplayBuffer rb):
        # appends s, pi, and z of rb
        if (self.s is None):
            self.s = rb.s
            self.pi = rb.pi
            self.z = rb.z
        else:
            self.s = np.concatenate((self.s, rb.s))
            self.pi = np.concatenate((self.pi, rb.pi))
            self.z = np.concatenate((self.z, rb.z))
    
    cpdef void mark_end(self, int iteration):
        # mark end of iteration to keep capacity amount of games
        cdef:
            int itr = iteration
            int length
            int idx
        
        length = self.s.shape[0] - np.sum(self.idx) if len(self.idx) > 0 \
                 else self.s.shape[0]
        if (len(self.idx) == self.capacity):
            self.idx.append(length)
            idx = self.idx.pop(0)
            self.s = self.s[idx:]
            self.pi = self.pi[idx:]
            self.z = self.z[idx:]
        else:
            self.idx.append(length)
    
    def sample(self, int num_epoch, int batch_size, int step_size):
        # sample num_epochs amount of batches, in partitions of step_size
        cdef:
            int epochs = num_epoch
            int bsize = batch_size
            int ssize = step_size
            int num_batches = self.s.shape[0] / bsize + 1
            np.ndarray s, pi, z
            int e, b, i
        
        e = 0
        while (e < epochs):
            b = 0
            while (b < num_batches):
                rand_idx = np.random.permutation(self.s.shape[0])[:bsize]
                s = self.s[rand_idx]
                pi = self.pi[rand_idx]
                z = self.z[rand_idx]
                i = 0
                while (i < bsize):
                    yield s[i:i+ssize], pi[i:i+ssize], z[i:i+ssize]
                    i += ssize
                b += 1
            e += 1
    
    cpdef void save(self, int i):
        # save replay buffer at iteration i
        np.save("{}/{:03d}/state{}.npy" \
                .format(self.path, i, "_td" if self.td else ""), self.s)
        np.save("{}/{:03d}/policy{}.npy" \
                .format(self.path, i, "_td" if self.td else ""), self.pi)
        np.save("{}/{:03d}/result{}.npy" \
                .format(self.path, i, "_td" if self.td else ""), self.z)
        np.save("{}/{:03d}/index{}.npy" \
                .format(self.path, i, "_td" if self.td else ""),
                np.array(self.idx, dtype=np.int64))
        print("ReplayBuffer saved.")
      
    cpdef void load(self, int i=-1):
        # load replay buffer at iteration i
        self.s = np.load("{}/{:03d}/state{}.npy" \
                         .format(self.path, i, "_td" if self.td else ""))
        self.pi = np.load("{}/{:03d}/policy{}.npy" \
                          .format(self.path, i, "_td" if self.td else ""))
        self.z = np.load("{}/{:03d}/result{}.npy" \
                         .format(self.path, i, "_td" if self.td else ""))
        self.idx.extend(np.load("{}/{:03d}/index{}.npy" \
                                .format(self.path, i, "_td" if self.td else "")))
        print("ReplayBuffer loaded.")

##############################################################################
##############################################################################



##############################################################################
####################### Monte Carlo Tree Search Logic ########################
##############################################################################

cdef str log_stats(Node node):
    cdef:
        vector[float] prior = node.m_prior
        vector[int] visit
        vector[float] value
        np.ndarray stats
        vector[string] index = deref(node.m_game).action_str()
        list columns = ["Prior", "Visit", "Value"]
        str state_repr = deref(node.m_game).state_str().decode('UTF-8')
        int i

    visit.reserve(prior.size())
    value.reserve(prior.size())
    i = 0
    while (i < prior.size()):
        visit.push_back(deref(node.m_children[i]).m_visit)
        value.push_back(1 - deref(node.m_children[i]).m_value)
        i += 1
    stats = np.stack((prior, visit, value)).T
    return "\n{}\n\nStatistics:\n{}\n".format(state_repr,
        pd.DataFrame(data=stats, index=index, columns=columns, dtype=np.float32) \
            .nlargest(10, columns="Visit"))

cdef np.ndarray[np.float32_t, ndim=4] tree_tonumpy(
        vector[vector[uint8_t]] states, int history,
        int state_depth, int board_size, int board_dim):
    # convert vectors generated from tree's get_eval_states to ndarrays
    cdef:
        np.ndarray[np.uint8_t, ndim=2] state_arr = np.empty(
            (states.size(), state_depth * board_size), dtype=np.uint8)
        uint8_t[::1] state_view
        int i
    
    i = states.size() - 1
    while (i >= 0):
        state_view = <uint8_t[:state_depth * board_size]> states[i].data()
        state_arr[i] = np.asarray(state_view)
        i -= 1
    return state_arr.reshape((states.size() / history, history * state_depth, 
                              board_dim, board_dim)).astype(np.float32)

cdef void search(vector[shared_ptr[Node]] &trees, list rbs, int sim_count,
                 int tau, eval_func, float c_puct, float alpha, float epsilon,
                 int cutoff, int history, float td, int save):
    # search trees and replace node with next game nodes
    cdef:
        vector[shared_ptr[Node]] chosen = trees
        np.ndarray[np.float32_t, ndim=1] priors, values
        vector[float] prior, value
        shared_ptr[Node] child
        vector[shared_ptr[Node]] children
        vector[uint16_t] actions
        np.ndarray[np.int_t, ndim=1] visit
        np.ndarray[np.float64_t, ndim=1] visit_soft
        np.ndarray[np.uint8_t, ndim=3] s
        np.ndarray[np.float32_t, ndim=1] pi
        float v = 0
        vector[uint8_t] state
        uint8_t[::1] state_view
        int board_dim = deref(deref(trees[0]).m_game).get_board_dim()
        int board_size = deref(deref(trees[0]).m_game).get_board_size()
        int ac_dim = deref(deref(trees[0]).m_game).get_ac_dim()
        int state_depth = deref(deref(trees[0]).m_game).get_state_depth()
        int idx
        int i, j
    
    # simulate
    i = sim_count
    while (i):
        select(trees, chosen, tau, c_puct, alpha, epsilon, cutoff)
        if (eval_func is not None):
            priors, values = eval_func(tree_tonumpy(
                get_eval_states(chosen, history, state_depth, board_size),
                history, state_depth, board_size, board_dim))
            # directly access array memory and assign to vector
            prior.assign(&priors[0], &priors[0] + trees.size() * ac_dim)
            value.assign(&values[0], &values[0] + trees.size())
        else:
            prior.assign(trees.size() * ac_dim, 1.0 / ac_dim)
            value = expand(chosen)
        backprop(trees, chosen, prior, value, ac_dim)
        i -= 1
    
    # choose next game by visit count
    i = trees.size() - 1
    while (i >= 0):
        child = trees[i]
        actions = deref(deref(child).m_game).get_action()
        visit = np.empty(actions.size(), dtype=np.int)
        j = actions.size() - 1
        while (j >= 0):
            visit[j] = deref(deref(child).m_children[j]).m_visit
            j -= 1
        idx = np.random.choice(actions.size(), p=visit/np.sum(visit)) if \
              (deref(deref(child).m_game).get_turn() <= cutoff) & (tau > 0) else \
              np.argmax(visit)
        if (td):
            v = deref(deref(child).m_children[idx]).m_value
        children = deref(child).m_children
        trees[i] = children[idx]
        # remove subtrees not chosen
        children.erase(children.begin(), children.begin() + idx)
        children.erase(children.begin() + 1, children.end())
        
        # save to replay_buffer
        if (save & (deref(deref(child).m_game).get_turn() >= state_depth)):
            state = deref(deref(child).m_game).get_state()
            # copies vector to array (prevent corruption)
            s = np.ascontiguousarray(state, dtype=np.uint8) \
                .reshape((state_depth, board_dim, board_dim))
            j = state_depth - 1
            while (j):
                child = deref(child).m_parent.lock()
                state = deref(deref(child).m_game).get_state()
                # directly access vector memory
                state_view = <uint8_t[:state.size()]> state.data()
                # since concatenate creates a copy anyways
                s = np.concatenate((np.asarray(state_view).reshape(
                    (state_depth, board_dim, board_dim)), s))
                j -= 1
            # copy softmaxed visit over legal actions to prior
            visit_soft = softmax(visit)
            pi = np.zeros(ac_dim, dtype=np.float32)
            j = actions.size() - 1
            while (j >= 0):
                pi[actions[j]] = visit_soft[j]
                j -= 1
            rbs[i].add_history(s, pi, v)
        i -= 1

cpdef (int, int, int) self_play(int a_ep_count, int a_sim_count, int a_tau,
                                eval_func_p1, eval_func_p2, ReplayBuffer rb,
                                float a_c_puct=2.5, float a_alpha=0.03,
                                float a_epsilon=0.25, int a_cutoff=30,
                                float a_td=0, int a_save=0, int a_log=0, file=None):
    # self play a_ep_count number of games, with a_sim_count number of simulations
    # per move, with a_tau indicating explore or not. p1 and p2 can have different
    # evaluation functions. a_c_puct, a_alpha, a_epsilon, and a_cutoff control
    # the extent of exploration. if a_save, then rb will store game data, with a_td
    # indicating the weight of td reward augmentation. if a_log, write game to file.
    # returns the number of wins for each player as tuples.

    cdef:
        int ep_count = a_ep_count
        int sim_count = a_sim_count
        int tau = a_tau
        float c_puct = a_c_puct
        float alpha = a_alpha
        float epsilon = a_epsilon
        int cutoff = a_cutoff
        int history = rb.history
        float td = a_td
        int save = a_save
        int log = a_log
        vector[shared_ptr[Node]] root_trees, trees
        int game_remain
        list rbs = None
        shared_ptr[Node] parent
        vector[int] scores
        int result
        str content = ""
        int i
    
    root_trees.reserve(ep_count)
    i = ep_count
    while (i):
        root_trees.push_back(make_shared[Node](make_shared[Game]()))
        i -= 1
    trees = root_trees
    if (save):
        rbs = [ReplayBuffer(td=td) for _ in range(ep_count)]
    scores.reserve(3)
    i = 3
    while (i):
        scores.push_back(0)
        i -= 1
    
    while (trees.size()):
        parent = trees[0]
        if (deref(deref(parent).m_game).get_turn() & 1):
            search(trees, rbs, sim_count, tau, eval_func_p1,
                   c_puct, alpha, epsilon, cutoff, history, td, save)
        else:
            search(trees, rbs, sim_count, tau, eval_func_p2,
                   c_puct, alpha, epsilon, cutoff, history, td, save)
        if (log):
            content += log_stats(deref(parent))
        i = trees.size() - 1
        while (i >= 0):
            if (deref(deref(trees[i]).m_game).terminal() == 0):
                result = deref(deref(trees[i]).m_game).evaluate()
                scores[result + 1] += 1
                trees.erase(trees.begin() + i)
                if (save):
                    rbs[i].update_result(result)
                    rb.extend(rbs.pop(i))
                if (log):
                    # file.write(content)
                    print(content)
            i -= 1
    return (scores[2], scores[0], scores[1]) # black, white, draw
