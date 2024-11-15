from setuptools import setup

setup(
    name='diffusion',
    version='1.0',
    description='Python module for solving diffusion equations using a shared C library',
    py_modules=['diffusion'],
    install_requires=[
        'numpy',
    ],
)