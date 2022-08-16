import io
import os

import setuptools

# Package meta-data.
NAME = "basalt2022"
DESCRIPTION = "Solution for Basalt 2022'"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/lauritowal/BASALT_2022.git"
EMAIL = ""
AUTHOR = ""
REQUIRES_PYTHON = ">=3.8.5"

# Package requirements.
base_packages = [
    'pyvirtualdisplay==3.0',
    'opencv-python==4.6.0.66',
    'matplotlib==3.5.2',
    'Pillow==9.2.0',
]
dev_packages = [
    "mypy==0.961",
    "pytest==7.1.2",
    "pylint==2.14.4",
    "black==22.6.0",
    "pip-tools==6.8.0"
]
docs_packages = []

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

setuptools.setup(
    name=NAME,
    version='0.0.1',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        #"test": dev_packages,
        "docs": docs_packages,
        "all": dev_packages + docs_packages
    },
    include_package_data=True,
    #license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)
