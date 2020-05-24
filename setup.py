from distutils.core import setup

setup(
    name='aqpolypy',
    version='0.5.0',
    author='A. Travesset',
    author_email='trvsst@ameslab.gov',
    packages=['towelstuff', 'towelstuff.test'],
    scripts=['bin/units_example.py','bin/water_properties_example.py'],
    url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Polymers, hydrogen bonds and electrolytes.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.18.1",
        "matplotlib >= 3.1.3",
        "sphinx >= 2.4.0",
        "scipy >= 1.4.1"
    ],
)
