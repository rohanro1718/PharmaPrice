
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of project requirements.
    """
    with open(file_path, "r") as file_obj:
        requirements = [
            req.strip()
            for req in file_obj.readlines()
            if req.strip()
        ]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="pharmaprice",
    version="0.0.1",
    author="Rohan",
    author_email="rohanro1718@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
