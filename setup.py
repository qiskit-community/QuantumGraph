from setuptools import setup, find_packages

setup(name='quantumgraph',
      install_requires=['qiskit', 'scipy', 'pairwise_tomography'],
      version='0.0.1',
      packages=[package for package in find_packages()
                if package.startswith('quantumgraph')]
)