import os

from setuptools import setup, find_packages

extras = {
    'jax': ['jax>=0.4.1',
            'jaxlib>=0.4.1',
            'optax>=0.1.4',
            'tqdm>=4.64.1',
            ],
    'pytorch': ['torch>=2.0.0',
                'tqdm>=4.64.1',
                ],
}
# Read in README.md for our long_description
cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='meent',
    version='0.11.0',
    url='https://github.com/kc-ml2/meent',
    author='KC ML2',
    author_email='yongha@kc-ml2.com',
    packages=['meent'] + find_packages(include=['meent.*']),
    install_requires=[
        'numpy>=1.23.3',
        'scipy>=1.9.1',
    ],
    extras_require=extras,
    python_requires='>=3.8',
    description=(
        "Electromagnetic simulation (RCWA) & optimization package in Python"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        'meent': ['nk_data/filmetrics/*.txt', 'nk_data/matlab/*.mat'],
    },
)
