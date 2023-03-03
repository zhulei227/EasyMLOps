from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read()

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='easymlops',
    version='0.1',
    python_requires='>=3.6',
    long_description=long_description,
    author='zhulei227',
    url='https://github.com/zhulei227/EasyMLOps',
    description='MLOps Toolkit In Pipeline',
    packages=find_packages(),
    license='Apache-2.0',
    install_requires=requirements)
