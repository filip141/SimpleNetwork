from distutils.core import setup

setup(
    name='simple_network',
    version='0.1',
    packages=['simple_network', 'simple_network.tools', 'simple_network.train', 'simple_network.layers'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='filip141',
    author_email='201134@student.pwr.wroc.pl',
    description='', requires=['tensorflow', 'numpy']
)
