import os
import setuptools


with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [pkg.strip() for pkg in f.readlines() if pkg.strip()]

__version__ = None
with open(os.path.join("astropandas", "version.py")) as f:
    exec(f.read())  # reads __version__ value

setuptools.setup(
    name="astropandas",
    version=__version__,
    author="Jan Luca van den Busch",
    description="Tools to expand on pandas functionality for astronomical operations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jlvdb/astropandas",
    packages=["astropandas"],
    install_requires=install_requires,
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3"])
