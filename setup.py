import sys
from setuptools import setup, find_packages


if sys.version_info[:2] < (3, 4):
    print("Python >= 3.4 is required.")
    sys.exit(-1)


setup(
    name="tbplot",
    version="0.0.1",
    description="A collection of plotting functions for tight-binding models",
    license="BSD",
    author="Dean Moldovan",
    author_email="dean0x7d@gmail.com",

    platforms=['Unix', 'Windows'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.9.0', 'scipy>=0.15', 'matplotlib>=1.5.0'],
    zip_safe=False,
)
