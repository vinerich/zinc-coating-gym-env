from setuptools import setup
from setuptools import find_packages
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')

setup(
    name='zinc_coating',
    version="0.1",
    description='',
    author='Robin Weinreich',
    author_email='robin.weinreich@posteo.de',
    download_url='',
    license='MIT',
    install_requires=[
        'gym==0.21',
    ],
    extras_require={

    },
    scripts=[],
    packages=find_packages()
)
