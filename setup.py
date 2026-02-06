"""Setup script for Firmus AI Factory Digital Twin."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for the long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="firmus-ai-factory",
    version="0.1.0",
    author="Dr. Daniel Kearney",
    author_email="daniel.kearney@firmus.co",
    description="Digital Twin framework for AI Factory infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniel-kearney/firmus-ai-factory",
    project_urls={
        "Bug Tracker": "https://github.com/daniel-kearney/firmus-ai-factory/issues",
        "Documentation": "https://github.com/daniel-kearney/firmus-ai-factory#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "mypy>=1.3.0",
        ],
        "optimization": [
            "pymoo>=0.6.0",
            "optuna>=3.1.0",
        ],
    },
)
