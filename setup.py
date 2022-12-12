from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# import prefetch_generator
# loading README
# long_description = prefetch_generator.__doc__

version_string = '1.0.12'

setup(
    name="petals_local_parallel",
    version=version_string,
    description="description",
    # long_description=long_description,

    # Author details
    author_email="yalisnyak@nes.com",
    url="https://github.com/BlackSamorez/petals_local_parallel",

    # Choose your license
    license='MIT',
    packages=find_packages(),

    classifiers=[
        # Indicate who your project is intended for
        # 'Development Status :: 5 - Production/Stable',
        # 'Intended Audience :: Science/Research',
        # 'Intended Audience :: Developers',
        # 'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        # 'License :: OSI Approved :: The Unlicense (Unlicense)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',

    ],

    # What does your project relate to?
    keywords='torch distributed, bloom',

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        # 'matplotlib==3.6.1',
        # "numpy==1.23.3",
        # "pandas==1.5.0",
        # "protobuf==3.20.3",
        # "scikit-learn==1.1.2",
        # "tokenizers==0.12.1",
        # "torch==1.12.1",
        # "torchgpipe==0.0.7",
        # "transformers==4.22.2",
    ],
)