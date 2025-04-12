from setuptools import setup, find_packages

setup(
    name="aerthos_quant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "statsmodels",
    ],
    author="Siyuan Liu",
    description="Quantitative trading & forecasting platform for Carbon Credit",
)
