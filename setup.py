"""
Setup script for GPT-style LLM project.
Enables `pip install -e .` for local development imports.
"""

from setuptools import setup, find_packages

setup(
    name="my_llm",
    version="0.1.0",
    description="GPT-style LLM trained from scratch with PyTorch",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "tokenizers>=0.15.0",
        "datasets>=2.14.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "sse-starlette>=1.6.5",
        "pydantic>=2.4.0",
        "pyyaml>=6.0.1",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.8.0",
        "ftfy>=6.1.1",
        "regex>=2023.10.3",
    ],
    entry_points={
        "console_scripts": [
            "llm-train=training.train:main",
            "llm-chat=cli.chat:main",
            "llm-server=api.server:main",
        ],
    },
)
