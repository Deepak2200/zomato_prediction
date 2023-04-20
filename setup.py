from setuptools import find_packages,setup
from typing import List

Hypen_e_dot="-e ."

def get_requiements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)

    return requirements






setup(
    name="Regressionproject2",
    version="0.0.1",
    author="Deepak",
    author_email="ydeepakdy123@gmial.com",
    install_requires=get_requiements("requirements.txt"),
    packages=find_packages()  #fuction is written list of package
)