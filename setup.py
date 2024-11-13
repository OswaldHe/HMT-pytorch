from setuptools import setup, find_packages

setup(
    name="hmt",
    version="0.1",
    packages=find_packages(),
    package_dir={
        "modeling_hmt": "modeling_hmt",
        "hmt_tools": "hmt_tools"
    }
)
