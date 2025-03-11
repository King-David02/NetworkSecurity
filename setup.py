from setuptools import setup, find_packages
ignore = '-e .'
def get_requirements(file):
    try:
        with open(file) as files:
            requirements = [req.strip() for req in files.readlines()]
            if ignore in requirements:
                requirements.remove(ignore)

        return requirements
    
    except FileNotFoundError:
        print("requirements.txt file not found")
        return []

setup(
    name="Network Security",
    version='0.0.1',
    author="Ajana King-David Ayomide",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
