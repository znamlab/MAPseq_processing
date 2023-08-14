from setuptools import setup, find_packages

setup(
    name="MAPseq_preprocess",
    version="v0.1",
    url="https://github.com/znamlab/MAPseq_processing",
    license="MIT",
    author="Znamenskiy lab",
    author_email="benita.turner-bridger@crick.ac.uk",
    description="Tools for processing MAPseq data",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git",
        "opencv-python",
        "bg_atlasapi",
        "seaborn",
        "znamutils @ git+ssh://git@github.com/znamlab/znamutils.git",
    ],
)
