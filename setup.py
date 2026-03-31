from setuptools import setup, find_packages

setup(
    name="synthipdata",
    version="0.1.0",
    author="Vineet Singh",
    author_email="",
    description="Synthetic Data Augmentation for Rare-Case IP Lifecycle Scenarios",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vineetsingh-vs/synthipdata",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "sentence-transformers>=2.2.0",
        "qdrant-client>=1.7.0",
        "boto3>=1.28.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
