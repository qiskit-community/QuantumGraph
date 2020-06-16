from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='quantumgraph',
      install_requires=requirements,
      version='0.0.1',
      packages=['quantumgraph']
)
