from setuptools import setup, find_packages

setup(
    name="diamond_predictor",
    version="0.1",
    packages=find_packages(where="."),  
    package_dir={"": "."},  
    install_requires=[
        "streamlit",
        "pandas",
        "scikit-learn",
        # Add other dependencies from requirements.txt
    ],
)