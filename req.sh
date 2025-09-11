#!/bin/bash

# Install packages
pip install -qq thop
pip install -qq albumentations

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
cp "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Datasets/Tehran_Mashhad/Tehran/Dataset.zip" /content/Data/

# Unzip only if not already done
if [ ! -d /content/Data/Dataset ]; then
    unzip -q "/content/Data/Dataset.zip" -d /content/Data/ || {
        echo "❌ Unzip failed!"
        exit 1
    }
else
    echo "Already unzipped 💁‍♂️"
fi

# Check the zip file
ls -lh "/content/Data/Dataset.zip"

# Compare checksums
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Datasets/Tehran_Mashhad/Tehran/Dataset.zip"
md5sum "/content/Data/Dataset.zip"

# Remove Colab default stuff
rm -rf /content/sample_data

# Go to your repo folder
cd "/content/TPCV-CD"
pwd
ls
