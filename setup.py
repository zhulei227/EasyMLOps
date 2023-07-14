from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read()

setup(
    name='easymlops',
    version='0.1.4',
    python_requires='>=3.6',
    author='zhulei227',
    description='MLOps Toolkit In Pipeline',
    packages=find_packages(),
    license='Apache-2.0',
    install_requires=requirements)
