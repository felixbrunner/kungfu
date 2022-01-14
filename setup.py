from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="kungfu",
    version="0.1",
    description="Toolbox for asset pricing in python/pandas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="felixbrunner",
    author_email="",
    url="https://github.com/path/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas",
    ],
    python_requires=">=3.6",
)
