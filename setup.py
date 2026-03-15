from setuptools import setup, find_packages
from M5_ML_Project.constant.training_pipeline import training_pipeline

def get_requirements(file_path):
    with open(file_path, "r") as file_obj:
        lines = [req.strip() for req in file_obj.readlines()]
        if "-e ." in lines:
            lines.remove("-e .")
        return lines
    
setup(
    name="M5 Forcaste ML Project",
    version="0.0.1",
    author="Rahul Kujur",
    author_email="rahulkjr9435@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements(training_pipeline.REQUIREMENTS_FILE_PATH)
)

