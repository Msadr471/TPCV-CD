#!/bin/bash

pip install -qq thop
pip install -qq albumentations

cd /content/
pwd

if [ -f /content/Data ]; then
    rm /content/Data
fi

mkdir -p /content/Data

cp "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Dataset.zip" /content/Data/
cp "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Tehran.zip" /content/Data/
cp "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Mashhad.zip" /content/Data/

cd /content/Data
pwd

# === Dataset unzip + md5 check ===
if [ ! -d /content/Data/Dataset ]; then
    unzip -q "Dataset.zip" || { echo "❌ Dataset unzip failed!"; exit 1; }
else
    echo "Dataset already unzipped 💁‍♂️"
fi
ls -lh "Dataset.zip"
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Dataset.zip"
md5sum "/content/Data/Dataset.zip"

# === Tehran unzip + md5 check ===
if [ ! -d /content/Data/Tehran ]; then
    unzip -q "Tehran.zip" || { echo "❌ Tehran unzip failed!"; exit 1; }
else
    echo "Tehran already unzipped 💁‍♂️"
fi
ls -lh "Tehran.zip"
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Tehran.zip"
md5sum "/content/Data/Tehran.zip"

# === Mashhad unzip + md5 check ===
if [ ! -d /content/Data/Mashhad ]; then
    unzip -q "Mashhad.zip" || { echo "❌ Mashhad unzip failed!"; exit 1; }
else
    echo "Mashhad already unzipped 💁‍♂️"
fi
ls -lh "Mashhad.zip"
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Mashhad.zip"
md5sum "/content/Data/Mashhad.zip"

rm -rf /content/sample_data

cd "/content/TPCV-CD"
pwd
ls
