# AlphaZero
----
C++ & Cython based efficient and extendable implementation of AlphaZero based on Deepmind's papers [Mastering the game of Go without Human Knowledge](https://deepmind.com/research/publications/mastering-game-go-without-human-knowledge) and [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://deepmind.com/research/publications/general-reinforcement-learning-algorithm-masters-chess-shogi-and-go-through-self-play). Tensorflow 2.x and PyTorch implementations available.

## Dependencies
* Python (>= 3.7.0)
* Cython (>=3.0a1)
* numpy (>=1.17.0)
* pandas (>=0.25.0)
* scipy (>=1.3.0)
* tensorflow (2.2.0)
* torch (1.5.0)

Install dependencies with conda environment (optional but strongly recommended) and pip.
```sh
$ conda create -n az_env python=3.7
$ source activate az_env
$ pip install -r requirements.txt
```

## Build Extension and Demo
To create the c++ and cython extensions, run ```python setup.py build``` which creates a build directory with the shared libraries ready to be imported. To get a sense of how to use the libraries, take a look at ```othello_demo.py``` or run ```
python othello_demo.py``` to see the code in action.

## Extend Existing Games
This project is built with extendability in mind. To run with custom game models:
1. Create game logic in ```games``` directory,  ```{game}.cpp``` and ```{game}.hpp``` files with the class name ```Game``` extending ```base::Game``` from ```"mcts.hpp"```. See ```games/othello.cpp``` and ```games/othello.hpp``` for example.
2. Create a copy of ```cy_mcts.pyx``` in ```mcts``` directory, replace ```cdef extern from "mcts.hpp" namespace "base"``` with ```cdef extern from "../games/{game}.hpp" namespace "{game}"```, and rename the pyx file as ```cy_{game}.pyx```. See ```mcts/cy_othello.pyx``` for example.
3. Append game name at the end of ```sources.txt```
4. Run ```python setup.py build``` to build the new extension. The ```build``` directory, ```cy_{game}.<platform>.so```, and  ```mcts/cy_{game}.cpp``` files should be created.
5. In a python script, ```from cy_{game} import self_play``` and call ```self_play``` 

The directory should look something like this:
```sh
AlphaZero
├── README.md
├── build
│   ├── lib.<platform>
│   │   ├── cy_{game}.<platform>.so
│   │   └── cy_othello.<platform>.so
│   └── temp.<platform>
│       ├── games
│       │   ├── {game}.o
│       │   └── othello.o
│       └── mcts
│           ├── cy_{game}.o
│           ├── cy_othello.o
│           └── mcts.o
├── cy_{game}.<platform>.so
├── cy_othello.<platform>.so
├── games
│   ├── {game}.cpp
│   ├── {game}.hpp
│   ├── othello.cpp
│   └── othello.hpp
├── mcts
│   ├── cy_mcts.pyx
│   ├── cy_{game}.cpp
│   ├── cy_{game}.pyx
│   ├── cy_othello.cpp
│   ├── cy_othello.pyx
│   ├── mcts.cpp
│   └── mcts.hpp
├── models
│   ├── model_tf.py
│   └── model_torch.py
├── othello_demo.py
├── requirements.txt
├── setup.cfg
├── setup.py
├── sources.txt
└── train.py
```

Note: simplifying step 2 is top priority; any suggestion is welcome.

## AlphaZero Training
To train on a game, run ```train.py``` with the following arguments:
* ```framework```: ```tf``` for TensorFlow or ```torch``` for PyTorch.
* ```game```: name of the game to train on
* ```board_dim```: board dimension of game
* ```ac_dim```: action dimension of game
* ```state_depth```: number of historical states * number of layers in state representation
* ```n_layers```: number of residual layers for network
* ```filters```: number of filters in residual layers
* ```head_filters```: number of filters in policy and value head
* ```sim_count```: number of simulations per move
* ```ep_count```: number of episodes to play per iteration
* ```epochs```: number of times to loop through replay buffer when updating
* ```batch_size```: batch size to sample from replay buffer
* ```step_size```: update batch in step size chunks
* ```buffer_size```: number of iterations to keep in replay buffer
* ```start_iter```: starting iteration; loads model and replay buffer if not zero
* ```n_iter```: number of iterations to train
* ```cpu```: train on CPU. Note: ```tf``` must be on GPU due to NCHW data format
* ```log_dir```: directory to save models and replay buffer to
