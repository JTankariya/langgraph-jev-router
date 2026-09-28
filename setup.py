from setuptools import setup, find_packages

setup(
    name="langgraph-jev-router",
    version="0.1.0",
    description="A plug-and-play System One conditional router for LangGraph using TypeSafe AI's Jev model.",
    author="Jayesh Tankariya",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
