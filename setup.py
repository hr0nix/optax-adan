from setuptools import setup

setup(
    name='optax-adan',
    version='0.1.3',
    description='An implementation of adan optimization algorithm for optax.',
    url='https://github.com/hr0nix/optax-adan',
    author='Boris Yangel',
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
