#!/bin/bash
# Make all Python scripts executable

chmod +x $(find "$(dirname "$0")" -name "*.py")

echo "All Python scripts are now executable!" 