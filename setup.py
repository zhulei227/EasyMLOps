from setuptools import find_packages, setup
import easymlops

with open('requirements.txt',encoding="utf8") as f:
    requirements = f.read()

setup(
    name='easymlops',
    version=easymlops.__version__,
    python_requires='>=3.6',
    author='zhulei227',
    description='MLOps Toolkit In Pipeline',
    packages=find_packages(),
    include_package_data=True,
    license='Apache-2.0',
    install_requires=requirements)
