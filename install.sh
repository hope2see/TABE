#!/bin/bash

OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"


# install miniconda -------------------------------------------------

if [[ "$OS" == "Linux" ]]; then
    if [[ "$ARCH" == "x86_64" ]]; then
        URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$ARCH" == "aarch64" ]]; then
        URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
elif [[ "$OS" == "Darwin" ]]; then  # macOS
    if [[ "$ARCH" == "x86_64" ]]; then
        URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    elif [[ "$ARCH" == "arm64" ]]; then  # Apple M1/M2/M3
        URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

wget -O miniconda.sh "$URL"
bash Miniconda3-latest-Linux-aarch64.sh
source ~/miniconda3/bin/activate
conda init


# Install git -----------------------------
sudo apt update
sudo apt install git -y

# Install gcc -----------------------------
sudo apt update
sudo apt install gcc -y

# Create python env ---------------------
conda create -n tabe1.0 python=3.11
conda activate tabe1.0


# Install tabe -------------------------------

echo "Downloading Tabe1.0 ..."
git clone https://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@github.com/hope2see/tabe1.0.git
cd tabe1.0

echo "Downloading pmdarima ..."
git clone https://github.com/alkaline-ml/pmdarima.git

echo "Downloading Time-Series-Library ..."
git clone https://github.com/thuml/Time-Series-Library.git

echo "Downloading Time-MoE ..."
git clone https://github.com/Time-MoE/Time-MoE.git


echo "Installing required packages for pmdarima ..."
pip install -r ./pmdarima/requirements.txt

echo "Installing required packages for Time-MoE ..."
pip install -r ./Time-MoE/requirements.txt

echo "Installing required packages for TABE ..."
pip install -r ./requirements.txt


# Build and install pmdarima -------------------------------
echo "Build & install pmdarima ..."
cd pmdarima; python3.11 setup.py develop; python3.11 setup.py install
