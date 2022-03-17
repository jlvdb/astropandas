import setuptools


with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt") as f:
    install_requires = [pkg.strip() for pkg in f.readlines() if pkg.strip()]

setuptools.setup(
    name="astropandas",
    version="1.0",
    author="Jan Luca van den Busch",
    description="Tools to expand on pandas functionality for astronomical operations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jlvdb/astropandas",
    packages=setuptools.find_packages(),
    install_requires=install_requires)
