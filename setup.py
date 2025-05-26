from setuptools import setup, find_packages

setup(
    name="oodgs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
    ],
    author="Pavle Padjin",
    description="OOD Gaussian Splatting",
    python_requires=">=3.8",
) 