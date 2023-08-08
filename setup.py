#!/usr/bin/env python3
#
# Please see pyproject.toml
#
# see: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "calibrated_explanations"
    version = "0.0.9a1"
    authors = [
    { name="Helena Löfström", email="helena.lofstrom@ju.se" },
    { name="Tuwe Löfström", email="tuwe.lofstrom@ju.se" },
    ]
    description = "Extract calibrated explanations from machine learning models."
    long_description=long_description,
    long_description_content_type="text/markdown",
    readme = {
        file = "README.md",
        content_type = "text/markdown",
    }
    dependencies = [
    'crepes',
    'ipython',
    'lime',
    'matplotlib',
    'numpy',
    'pandas',
    'scikit-learn',
    'shap',
    ]
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numpy", "pandas", "scikit-learn", "lime", "crepes", "matplotlib"],
    python_requires=">=3.8",
)
