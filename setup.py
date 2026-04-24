from setuptools import setup, find_packages

setup(
    name="seasonal-analysis-pipeline",
    version="0.1.0",
    description="End-to-end data engineering pipeline for seasonla analyisis of foinancila assets",
    author="Alejandro Maza V.",
    packages=find_packages(where="src",exclude=("tests", "notebooks")),
    package_dir={"": "src"},              # map root to src/
    include_package_data=True,
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.9",
)