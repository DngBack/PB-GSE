"""
Setup script for PB-GSE
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pb-gse",
    version="1.0.0",
    author="PB-GSE Team",
    author_email="your-email@example.com",
    description="PAC-Bayes Group-Selective Ensemble for Long-tail Classification with Abstention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/PB-GSE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pb-gse-train=pb_gse.scripts.run_experiment:main",
            "pb-gse-demo=pb_gse.scripts.demo:main",
            "pb-gse-ablation=pb_gse.scripts.run_ablation:main",
        ],
    },
)
