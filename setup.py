from setuptools import setup

setup(
    name='diffusion',
    version='3.1',
    description='Python module for solving diffusion equations using a shared C library',
    py_modules=['diffusion'],
    install_requires=[
        'numpy',
    ],
)