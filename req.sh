#!/bin/bash

# Install packages
# pip install tensorboardX==2.6.2.2
pip install tensorboardX
pip install thop

# Print working dir
pwd
cd /content/
pwd

# Fix /content/Data if it's a file
if [ -f /content/Data ]; then
    rm /content/Data
fi

# Make sure /content/Data is a directory
mkdir -p /content/Data

# Copy zip file
cp "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Datasets/3DCD/3DCD.zip" /content/Data/

# Unzip only if not already done
if [ ! -d /content/Data/3DCD ]; then
    unzip /content/Data/3DCD.zip -d /content/Data/
else
    echo "Already unzipped 💁‍♂️"
fi

# Check the zip file
ls -lh /content/Data/3DCD.zip

# Compare checksums
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Datasets/3DCD/3DCD.zip"
md5sum /content/Data/3DCD.zip

# Remove Colab default stuff
rm -rf /content/sample_data

# Go to your repo folder
cd /content/USSFC-Net
pwd
ls
