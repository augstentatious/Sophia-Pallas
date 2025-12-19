from setuptools import setup, find_packages

setup(
    name="unswag",
    version="0.1.0",
    description="High-integrity, memory-efficient JAX kernels for Gemma-9B fine-tuning.",
    author="John Augustine Young",
    author_email="cleanroomresearch@gmail.com", # ;)
    packages=find_packages(),
    install_requires=[
        "jax[tpu]",
        "flax",
        "numpy",
        "requests", # Needed for Kaggle TPU handshake
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
