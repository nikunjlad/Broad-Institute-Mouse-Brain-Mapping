#!/bin/bash

# initial centos update
echo "Updating Centos..."
sudo yum update -y

if [[ $? -eq 0 ]]; then
    echo "System updated successfully!"
else
    echo "System updates failed!"
    exit 1
fi

# install wget
echo "Installing wget..."
sudo yum install wget -y

if [[ $? -eq 0 ]]; then
    echo "wget installed successfully!"
else
    echo "wget installation failed!"
    exit 1
fi

# install git
echo "Installing Git..."
sudo yum install git -y

if [[ $? -eq 0 ]]; then
    echo "Git installed successfully!"
else
    echo "Git installation failed!"
    exit 1
fi

# install nano
echo "Installing Nano editor..."
sudo yum install git -y

if [[ $? -eq 0 ]]; then
    echo "Nano editor installed successfully!"
else
    echo "Nano editor installation failed!"
    exit 1
fi

# Python installation
cd ~
echo "Installing Python pre-requisites..."
sudo yum install gcc openssl-devel bzip2-devel libffi-devel -y

if [[ $? -eq 0 ]]; then
    echo "Python pre-requisites installed successfully!"
else
    echo "Python pre-requisites installation failed!"
    exit 1
fi

# changing to source directory for python tar installation and downloading the tar file
cd /usr/src
echo "Downloading Python..."
sudo wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz

if [[ $? -eq 0 ]]; then
    echo "Python installation tar downloaded successfully!"
else
    echo "Python installation tar download failed!"
    exit 1
fi

# extracting the tar file
echo "Extracting Python from tar..."
sudo tar xzf Python-3.7.3.tgz

if [[ $? -eq 0 ]]; then
    echo "Tar extraction completed successfully!"
else
    echo "Tar extraction failed!"
    exit 1
fi

# configuring python installation
cd Python-3.7.3
echo "Configuring Python..."
sudo ./configure --enable-optimizations

if [[ $? -eq 0 ]]; then
    echo "Python configuration completed successfully!"
else
    echo "Python configuration failed!"
    exit 1
fi

# building python from source
echo "Building Python from source..."
sudo make altinstall

if [[ $? -eq 0 ]]; then
    echo "Building Python from source completed successfully!"
else
    echo "Building Python from source failed!"
    exit 1
fi

# removing the directory
echo "Deleting Python installation tar..."
sudo rm /usr/src/Python-3.7.3.tgz

if [[ $? -eq 0 ]]; then
    echo "Python installation tar deleted successfully!"
else
    echo "Python installation tar deletion failed!"
    exit 1
fi

cd ~ && cd /usr/local/bin

echo "Generating symbolic links for python and pip..."
sudo ln -s python3.7 python3
sudo ln -s pip3.7 pip

if [[ $? -eq 0 ]]; then
    echo "Symbolic links for pip and python created successfully!"
else
    echo "Symbolic link creation for pip and python failed!"
    exit 1
fi

# updating pip installation
echo "Updating pip..."
cd ~
pip install --upgrade pip --user

if [[ $? -eq 0 ]]; then
    echo "Pip updated successfully!"
else
    echo "Pip updation failed!"
    exit 1
fi

mkdir envs && cd envs

# creating virtual environment for python
echo "Creating Python virtual environment..."
pip install virtualenv
virtualenv venv

if [[ $? -eq 0 ]]; then
    echo "Virtual environment created successfully!"
else
    echo "Virtual environment creation failed!"
    exit 1
fi



