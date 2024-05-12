import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="marl_dominoes",
    version="0.0.0",
    author="Bruno Zorrilla",
    author_email="brunozm92@gmail.com",
    description="Reproducible research for multi-agent reinforcement learning in the game of dominoes.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ebezzam/python-dev-tips",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "open_spiel",
        "numpy",
        "scipy",
        "matplotlib",
        "hydra-core",
        "tqdm",
    ],
    include_package_data=True,
)