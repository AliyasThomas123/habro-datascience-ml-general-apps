from setuptools import setup , find_packages
HYPHEN_E_DOT = "-e ."

def get_requirements(file_path):
    '''
    function for fetching requirements
    from requirements.txt file 
    it will fetch all the required libraries

    '''
    with open(file_path) as obj:
        requirements=obj.readlines()
        requirements = [req.replace("\n" , "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        return requirements



setup(

   name= "DataScience ML App General" ,
   version= "0.0.1" ,
   author="Aliyas Thomas",
   author_email="aliazthomaz@gmail.com" ,
   description="General application structure for Data science applications",
   packages=find_packages(),
   install_requires=get_requirements("requirements.txt")

)