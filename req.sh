#!/bin/bash

pip install -qq thop
pip install -qq albumentations

pwd
cd /content/
pwd

if [ -f /content/Data ]; then
    rm /content/Data
fi

mkdir -p /content/Data

cp "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Datasets/Tehran_Mashhad/Tehran/Dataset.zip" /content/Data/

cd /content/Data
pwd

if [ ! -d /content/Data/Dataset ]; then
    unzip -q "Dataset.zip" -d /content/Data/ || {
        echo "❌ Unzip failed!"
        exit 1
    }
else
    echo "Already unzipped 💁‍♂️"
fi

ls -lh "Dataset.zip"

md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Datasets/Tehran_Mashhad/Tehran/Dataset.zip"
md5sum "/content/Data/Dataset.zip"

rm -rf /content/sample_data

cd "/content/TPCV-CD"
pwd
ls
