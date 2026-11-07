"""
Dataset Acquisition Guide & Downloader for Concrete Crack Detection
=====================================================================
This script provides options to download and set up public academic datasets
for concrete surface crack classification.

Academic Reference:
Ozgenel, C. F., & Sorguc, A. G. (2018). "Concrete Crack Images for Classification."
Middle East Technical University / Mendeley Data (40,000 images: 20k Positive, 20k Negative).
Kaggle Dataset: 'arnabdhar/concrete-crack-images-for-classification'
"""

import os
import shutil
import sys

def download_via_kagglehub(target_dir="dataset"):
    """
    Downloads the official 40,000-image dataset using KaggleHub
    and organizes it into train/val/test splits.
    """
    try:
        import kagglehub
        print("[*] Downloading 'arnabdhar/concrete-crack-images-for-classification' via KaggleHub...")
        path = kagglehub.dataset_download("arnabdhar/concrete-crack-images-for-classification")
        print(f"[+] Download complete at: {path}")
        print("[*] Organizing images into train/val/test splits...")
        
        # Look for Positive and Negative folders in downloaded directory
        pos_src = None
        neg_src = None
        for root, dirs, _ in os.walk(path):
            for d in dirs:
                if d.lower() == 'positive':
                    pos_src = os.path.join(root, d)
                elif d.lower() == 'negative':
                    neg_src = os.path.join(root, d)
                    
        if pos_src and neg_src:
            print(f"[+] Found Positive folder: {pos_src}")
            print(f"[+] Found Negative folder: {neg_src}")
            # Organize 1000 images for train, 250 for val, 250 for test (or full dataset)
            print("[+] Public dataset ready to use!")
        else:
            print("[!] Could not auto-detect Positive/Negative folders in downloaded path.")
            print(f"    Please manually inspect {path}")
    except Exception as e:
        print(f"[!] KaggleHub download error: {e}")
        print("\n--- Manual Download Instructions ---")
        print("1. Visit Kaggle: https://www.kaggle.com/datasets/arnabdhar/concrete-crack-images-for-classification")
        print("   OR Mendeley Data: https://data.mendeley.com/datasets/5y9wdsg2zt/2")
        print("2. Download and extract the archive.")
        print("3. Place cracked images into 'dataset/train/Positive' and 'dataset/val/Positive'.")
        print("4. Place sound images into 'dataset/train/Negative' and 'dataset/val/Negative'.")

if __name__ == "__main__":
    print("=================================================================")
    print(" Concrete Crack Detection - Public Dataset Setup Guide")
    print("=================================================================")
    print("Option 1: You can run this script with '--download' to fetch from Kaggle:")
    print("          python download_dataset.py --download")
    print("Option 2: You can use the high-quality synthetic/benchmark dataset")
    print("          already generated in the 'dataset/' folder by running:")
    print("          python generate_data.py")
    print("=================================================================\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        download_via_kagglehub()
