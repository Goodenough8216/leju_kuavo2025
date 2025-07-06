#!/bin/bash

# Exit on error
set -e

# Script to generate documentation for Kuavo Humanoid SDK

# Install required packages
echo "Installing required packages..."
pip list | grep -E "sphinx|sphinx-rtd-theme|sphinx-markdown-builder" > /dev/null || pip install sphinx sphinx-rtd-theme sphinx-markdown-builder

# Check if ROS environment is set up
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS environment not found. Please source your ROS setup file first."
    echo "Example: source /opt/ros/<distro>/setup.bash"
    exit 1
fi

# Check if humanoid_controllers is running
if ! pgrep -f "roslaunch humanoid_controllers" > /dev/null; then
    echo "Error: humanoid_controllers not running. Please start it first."
    echo "Example: roslaunch humanoid_controllers humanoid_controllers.launch"
    exit 1
fi


echo "Generating HTML documentation..."
sphinx-build -b html docs/ docs/html/
echo "HTML documentation generated successfully in docs/html/"

echo -e "\nGenerating Markdown documentation..."
sphinx-build -b markdown ./docs/ ./docs/markdown/
echo "Markdown documentation generated successfully in docs/markdown/"