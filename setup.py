from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read()
setup(
    name='pipeml',
    version='0.0.1',
    python_requires='>=3.6',
    author='zhulei227',
    url='https://github.com/zhulei227/pipeml',
    description='ML Pipeline Toolkit',
    packages=find_packages(),
    license='Apache-2.0',
    install_requires=requirements)
