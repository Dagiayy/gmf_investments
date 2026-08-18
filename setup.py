from setuptools import setup, find_packages

setup(
    name="gmf_investments",
    version="1.0.0",
    description="Financial Data Analytics, Volatility Modeling, Time Series Forecasting, Portfolio Optimization, and Backtesting Pipeline",
    author="Dagmawi Ayenew",
    author_email="ayenewdagmawi@gmail.com",
    url="https://github.com/Dagiayy/gmf_investments",
    packages=find_packages(),
    py_modules=["main"],
    install_requires=[
        "pandas>=1.4.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "python-docx>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "gmf-investments=main:run_pipeline",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
