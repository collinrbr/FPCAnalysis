from setuptools import setup, find_packages

setup(
    name='FPCAnalysis',
    version='1.0.0',
    author='Collin Brown',
    author_email='collin.crbrown@gmail.com',
    description='A library for computing the FPC',
    url='https://github.com/collinrbr/FPCAnalysis',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'requests'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
	'Topic :: Scientific/Engineering :: Physics :: Plasma Physics',
	'Topic :: Scientific/Engineering',
        'Topic :: Physics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Physics :: Plasma Physics',
        'Topic :: Scientific/Engineering :: Physics :: Data Analysis',
        'Topic :: Scientific/Engineering :: Physics :: Simulation',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    python_requires='==3.11'
)
