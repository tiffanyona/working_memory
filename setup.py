from setuptools import setup, find_packages

setup(
    name="working_memory",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[  # List your dependencies here
        "numpy",
        "pandas",
        "elephant",
        "neo",
        "sklearn",
        "matplotlib",
        "seaborn"
        "statsmodels",
        "scikit-learn"
        
    ],
)
