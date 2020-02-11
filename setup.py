import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="tess_cpm",
    version="0.0.1",
    author="Soichiro Hattori",
    author_email="soichiro@nyu.edu",
    url="https://github.com/soichiro-hattori/tess_cpm",
    license="MIT",
    description="An implementation of the Causal Pixel Model (CPM) for TESS data",
    long_description="readme",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)
