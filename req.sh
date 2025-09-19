#!/bin/bash

# pip install -qq thop
pip install -qq tensorboardX
pip install -qq albumentations

cd /content/
pwd

if [ -f /content/Data ]; then
    rm /content/Data
fi

mkdir -p /content/Data

# Function to copy file only if it doesn't exist or MD5 differs
copy_if_different() {
    local source_file="$1"
    local dest_file="$2"
    
    if [ ! -f "$dest_file" ]; then
        echo "📋 Copying $source_file (file doesn't exist)"
        cp "$source_file" "$dest_file"
    else
        local source_md5=$(md5sum "$source_file" | cut -d' ' -f1)
        local dest_md5=$(md5sum "$dest_file" | cut -d' ' -f1)
        
        if [ "$source_md5" != "$dest_md5" ]; then
            echo "🔄 Copying $source_file (MD5 differs: $source_md5 vs $dest_md5)"
            cp "$source_file" "$dest_file"
        else
            echo "✅ $dest_file already exists and matches source - skipping copy"
        fi
    fi
}

# Copy files only if needed
copy_if_different "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Dataset.zip" "/content/Data/Dataset.zip"
copy_if_different "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Tehran.zip" "/content/Data/Tehran.zip"
copy_if_different "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Mashhad.zip" "/content/Data/Mashhad.zip"

cd /content/Data
pwd

# Function to unzip only if directory doesn't exist
safe_unzip() {
    local zip_file="$1"
    local target_dir="$2"
    
    if [ ! -d "$target_dir" ]; then
        echo "📦 Unzipping $zip_file"
        unzip -q "$zip_file" || { echo "❌ $zip_file unzip failed!"; exit 1; }
    else
        echo "✅ $target_dir already exists - skipping unzip"
    fi
}

# === Dataset unzip ===
safe_unzip "Dataset.zip" "/content/Data/Dataset"

# === Tehran unzip ===
safe_unzip "Tehran.zip" "/content/Data/Tehran"

# === Mashhad unzip ===
safe_unzip "Mashhad.zip" "/content/Data/Mashhad"

# Optional: Verify MD5 checksums for debugging
echo "🔍 Verifying MD5 checksums:"
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Dataset.zip"
md5sum "/content/Data/Dataset.zip"
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Tehran.zip"
md5sum "/content/Data/Tehran.zip"
md5sum "/content/drive/MyDrive/GoogleColabDrive/CustomDataset/Mashhad.zip"
md5sum "/content/Data/Mashhad.zip"

rm -rf /content/sample_data

cd "/content/TPCV-CD"
pwd
ls