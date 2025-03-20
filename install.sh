# git clone https://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@github.com/hope2see/tabe1.0.git
# %cd tabe1.0/

#!/bin/bash

# Get the current Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')

# Define the required version
REQUIRED_VERSION="3.11.11"

# Install python3.11 if not installed
if [[ "$PYTHON_VERSION" != "$REQUIRED_VERSION" ]]; then
    echo "Python version($REQUIRED_VERSION) is required. So, install it ..."
    sudo apt-get update -y
    sudo apt-get install python3.11
    sudo apt install python3.11-dev
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
fi

echo "Downloading pmdarima ..."
git clone https://github.com/alkaline-ml/pmdarima.git

echo "Downloading Time-Series-Library ..."
git clone https://github.com/thuml/Time-Series-Library.git

echo "Downloading Time-MoE ..."
git clone https://github.com/Time-MoE/Time-MoE.git


echo "Installing pmdarima ..."
pip install -r ./pmdarima/requirements.txt
OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
    sed -i '' 's/setuptools>=38.6.0,!=50.0.0/setuptools==58.0.4/' ./pmdarima/requirements.txt
elif [[ "$OS" == "Linux" ]]; then
    sed -i 's/setuptools>=38.6.0,!=50.0.0/setuptools==58.0.4/' ./pmdarima/requirements.txt
else
    echo "Unsupported OS: $OS"
fi
cd pmdarima; python3.11 setup.py develop; python3.11 setup.py install


echo "Installing required packages for Time-MoE ..."
pip install -r ./Time-MoE/requirements.txt

echo "Installing required packages for TABE ..."
pip install -r ./requirements.txt

