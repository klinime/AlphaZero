
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path

path = Path(__file__).parent
abspath = str(path.absolute())
with open(path/'sources.txt', 'r') as file:
	extensions = [
		Extension('cy_' + s,
			['{}/mcts/cy_{}.pyx'.format(abspath, s),
			 '{}/games/{}.cpp'.format(abspath, s),
			 '{}/mcts/mcts.cpp'.format(abspath)],
			extra_compile_args=['-std=c++17'])
		for s in file.read().splitlines()
	]
	setup(
		name='cy_mcts',
		version='1.0',
		ext_modules=cythonize(extensions),
		include_dirs=[np.get_include()],
		author='Kevin Lin',
		author_email='klinime@gmail.com',
		description='Setup extensions for AlphaZero MCTS',
	)
