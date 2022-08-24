from setuptools import setup
from pathlib import Path
import optax_adan.version as version

cur_dir = Path(__file__).parent
long_description = (cur_dir / "README.md").read_text()

setup(
    name='optax-adan',
    version=version.__version__,
    description='An implementation of adan optimization algorithm for optax.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hr0nix/optax-adan',
    author=version.__author__,
    author_email='boris.jangel@gmail.com',
    license='Apache License 2.0',
    packages=['optax_adan'],
    install_requires=['optax'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
