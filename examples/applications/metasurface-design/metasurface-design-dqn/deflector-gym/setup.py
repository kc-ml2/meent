from setuptools import setup, find_packages

setup(
    name='deflector_gym',
    version='0.0.0',
    url='https://github.com/kc-ml2/deflector-gym',
    author='KC ML2',
    author_email='anthony@kc-ml2.com',
    packages=find_packages(),
    install_requires=[
        'gym<0.25',
        'meent==0.3.8',
        'tqdm',
    ],
    python_requires='>=3.8'
)
