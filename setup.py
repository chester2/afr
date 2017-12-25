from setuptools import setup, find_packages

setup(
    name='afr',
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'Pillow'],
    author='Chester Wu',
    author_email='chester.wx.wu@gmail.com',
    license='MIT',
    url='https://github.com/chester2/afr'
)