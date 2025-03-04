from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aioredlock",
    version="0.8.0",
    description="Asyncio implementation of Redis distributed locks",
    long_description=long_description,
    url="https://github.com/joanvila/aioredlock",
    author="Joan Vilà Cuñat",
    author_email="vila.joan94@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="redis redlock distributed locks asyncio",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["redis>=5.0.0", "attrs >= 17.4.0"],
    extras_require={
        "test": [
            "pytest<=8.0.0",
            "pytest-asyncio",
            "pytest-mock",
            "pytest-cov",
            "flake8",
        ],
        "cicd": ["codecov"],
        "package": ["bump2version", "twine", "wheel"],
        "examples": ["aiodocker"],
    },
)
