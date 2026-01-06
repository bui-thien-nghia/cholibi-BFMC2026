from setuptools import find_packages, setup

package_name = 'bfmc_hardware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kyrgios',
    maintainer_email='khoi.vuminh241207@hcmut.edu.vn',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = bfmc_hardware.camera_node:main',
            'serial_node = bfmc_hardware.serial_node:main'
        ],
    },
)
