from setuptools import setup, find_packages

setup(
    name='meent',
    version='0.4.2',
    url='https://github.com/kc-ml2/meent',
    author='KC ML2',
    author_email='yongha@kc-ml2.com',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.3',
        'scipy==1.9.1',
        'jax==0.3.21',
        'matplotlib==3.5.3',
    ],
    python_requires='>=3.8',
    long_description_content_type="text/markdown",
    package_data={
        'meent': ['nk_data/filmetrics/*.txt', 'nk_data/matlab/*.mat'],
    },
)
