"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""


from setuptools import setup

setup(
    name="segthraws",
    version="1.0.0",
    description="Segmentation of Thermal Hotspots in Raw Sentinel2 data (SegTHRawS) is an open-source Python package"
    + " that provides a comprehensive set of tools for creating a segmentation dataset and training DL segmentation models with Sentinel-2 Raw data. ",
    # long_description=open("README.md", encoding="cp437").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/Ubotica/SegTHRaws",
    author="Cristopher Castro Traba",
    author_email="cristophercastrotraba7@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["segthraws","pyraws", "pyraws.raw", "pyraws.database", "pyraws.l1", "pyraws.utils"],
    python_requires=">=3.10, <4",
    project_urls={"Source": "https://github.com/Ubotica/SegTHRawS"},
)
