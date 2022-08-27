from setuptools import setup, find_packages

setup(
    name='typeform_extractor',
    version='0.9.9.2',
    license='MIT',
    author="Daniel Vivas",
    author_email='hello@danielvivas.com',
    url='https://github.com/danielvivaspe/typeform-extractor',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    keywords='typeform sentiment analysis dataframe amazon aws',
    install_requires=['boto3', 'requests', 'pandas', 'formulas', 'importlib_metadata'],
    python_requires='>=3.7',
    description='Simple package to make data extraction from Typeform easier, analyze sentiment with AWS and calculate metrics with the data'
)
