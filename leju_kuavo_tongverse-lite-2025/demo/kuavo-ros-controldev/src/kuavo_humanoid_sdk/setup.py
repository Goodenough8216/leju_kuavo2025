from setuptools import setup

with open("sdk_description.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kuavo_humanoid_sdk",
    license="MIT",
    author=["lejurobot"],
    author_email=["edu@lejurobot.com"],
    version=open("VERSION", "r").read().strip(),
    description="A Python SDK for kuavo humanoid robot.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/leju-robot/kuavo-ros-opensource/",
    keywords=["kuavo", "humanoid", "robot", "robotics", "lejurobot", "ros"],
    packages=[
    'kuavo_humanoid_sdk',
    'kuavo_humanoid_sdk.common',
    'kuavo_humanoid_sdk.interfaces',
    'kuavo_humanoid_sdk.kuavo',
    'kuavo_humanoid_sdk.kuavo.core',
    'kuavo_humanoid_sdk.kuavo.core.ros',
    ],
    install_requires=[
        "numpy", 
        "transitions",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta", 
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Documentation": "https://gitee.com/leju-robot/kuavo-ros-opensource/",
        "Source Code": "https://gitee.com/leju-robot/kuavo-ros-opensource/",
    }
)