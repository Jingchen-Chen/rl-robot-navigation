from setuptools import setup, find_packages
import os

def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rl-robot-navigation",
    version="0.1.0",
    author="Jingchen Chen",
    description="Deep Reinforcement Learning for Robot Navigation",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Jingchen-Chen/rl-robot-navigation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
