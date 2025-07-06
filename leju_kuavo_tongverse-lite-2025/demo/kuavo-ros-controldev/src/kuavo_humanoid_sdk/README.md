

# Kuavo Humanoid SDK

A comprehensive Python SDK for controlling Kuavo humanoid robots. This SDK provides interfaces for robot state management, arm and head control, and end-effector operations. It is designed to work with ROS (Robot Operating System) environments.

**Warning**: This SDK currently only supports **ROS1**. ROS2 support is not available.

PyPI Package: https://pypi.org/project/kuavo-humanoid-sdk/

## Installation

To install Kuavo Humanoid SDK, you can use pip:
```bash
 pip install kuavo-humanoid-sdk
```

For development installation (editable mode), use:
```bash
 pip install -e .
```

## Description

For detailed SDK documentation and usage examples, please refer to [sdk_description.md](sdk_description.md).

## Documentation

The documentation is available in two formats:
- HTML format: [docs/html](docs/html)
- Markdown format: [docs/markdown](docs/markdown)


## For Maintainers

### Package the SDK
### Version Status Levels

When updating the `setup.py` classifiers, use the appropriate Development Status:
- `5 - Production/Stable `
- `4 - Beta `

To package the SDK for distribution, follow these steps:
```bash
python3 setup.py sdist bdist_wheel

ls -lh ./dist/
总用量 76K
-rw-rw-r-- 1 lab lab 41K 2月  22 17:37 kuavo_humanoid_sdk-0.1.0-py3-none-any.whl
-rw-rw-r-- 1 lab lab 32K 2月  22 17:37 kuavo_humanoid_sdk-0.1.0.tar.gz
```

### Upload to PyPI
First, create or edit `~/.pypirc` with your API token:
```
[testpypi]
username = __token__
password = pypi-<your-token>
```

To upload the package to PyPI, use twine:
```bash
pip install --upgrade requests-toolbelt
pip install "urllib3<=1.26.16" "twine<4.0.0" pyopenssl cryptography

twine upload --repository testpypi dist/* --verbose
```