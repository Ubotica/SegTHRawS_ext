from setuptools import find_packages, setup

setup(
    name='SegTHRawS',
    packages=find_packages(include=['simera_processing']),
    version='1.0.0',
    description='Segmentation of Thermal Hotspots on Raw Sentinel-2 imagery',
    author='Cristopher Castro Traba',
    author_email=['cristopher.traba@ubotica.com'],
)

