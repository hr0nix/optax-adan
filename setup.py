from setuptools import setup
from pathlib import Path

_CURRENT_DIR = Path(__file__).parent


def _get_long_description():
    return (_CURRENT_DIR / "README.md").read_text()


def _get_version():
    with open(_CURRENT_DIR / 'optax_adan' / '__init__.py', 'r') as init_file:
        for line in init_file:
            if line.startswith('__version__') and '=' in line:
                version = line[line.find('=') + 1:].strip(' \'"\n')
        if version:
            return version
        raise ValueError('`__version__` not defined in `optax_adan/__init__.py`')


setup(
    name='optax-adan',
    version=_get_version(),
    description='An implementation of adan optimization algorithm for optax.',
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/hr0nix/optax-adan',
    author='Boris Yangel',
    author_email='boris.jangel@gmail.com',
    license='Apache License 2.0',
    packages=['optax_adan'],
    install_requires=['optax'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
