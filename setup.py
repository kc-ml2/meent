from setuptools import setup, find_packages

setup(
    name='meent',
    version='0.7.5',
    url='https://github.com/kc-ml2/meent',
    author='KC ML2',
    author_email='yongha@kc-ml2.com',
    packages=['meent'] + find_packages(include=['meent.*']),
    install_requires=[
        'numpy==1.23.3',
        'jax==0.4.1',
        'matplotlib==3.5.3',
    ],
    python_requires='>=3.8',
    long_description_content_type="text/markdown",
    package_data={
        'meent': ['nk_data/filmetrics/*.txt', 'nk_data/matlab/*.mat'],
    },
)
