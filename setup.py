
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

with open('./sources.txt', 'r') as file:
    extensions = [
        Extension('cy_' + s,
            ['./mcts/cy_{}.pyx'.format(s), './games/{}.cpp'.format(s), './mcts/mcts.cpp'],
            extra_compile_args=['-std=c++17'],
            include_dirs=[np.get_include()])
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
