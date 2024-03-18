from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "phonax/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="phonax",
    version=version,
    description="Phonax is an open-source code for predicting and training with second derivative Hessians of the E(3) equivariant neural network energy model",
    download_url="https://github.com/atomicarchitects/phonax",
    author="Shiang Fang, Mario Geiger, Joseph Checkelsky, Tess Smidt",
    python_requires=">=3.9",
    packages=find_packages(include=["phonax", "phonax.*"]),
    package_data={'':['*.json','*.pkl','*.yaml']},
    include_package_data=True,
    install_requires=[
        "numpy",
        "jax",
        "e3nn_jax>=0.19.3",
        "jraph",
        "tqdm",
        "dm-haiku",
        "ase",
        "matscipy",
        "roundmantissa",
        "spglib",
        "pymatgen",
        "optax",
        "matplotlib",
        "pandas",
        "pyyaml",
        "gradio",
        "unique_names_generator",
    ],
    #zip_safe=True,
)
