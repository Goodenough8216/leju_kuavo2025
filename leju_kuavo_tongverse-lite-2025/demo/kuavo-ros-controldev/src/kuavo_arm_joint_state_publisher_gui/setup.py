from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['kuavo_arm_joint_state_publisher_gui'],
    package_dir={'': 'src'}
)

setup(**d)
