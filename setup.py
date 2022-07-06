import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="Graph_RL",
    version="0.1.2",
    description="Hierarchical reinforcement learning framework which uses a directed graph to define the hierarchy.",
    author="Nico Gürtler",
    author_email="nico.guertler@tuebingen.mpg.de",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/nicoguertler/graph_rl",
    keywords=["reinforcement learning", "hierarchical reinforcement learning"],
    packages=find_packages(),
    install_requires=["numpy", "gym==0.21.0", "tianshou==0.3.1", "pyglet", "dyn_rl_benchmarks"]
)

