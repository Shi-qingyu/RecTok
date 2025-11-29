"""
Script to compress folders in work_dirs/tokenizer_training and upload to HuggingFace.
Each folder will be compressed into a zip file and uploaded to the repository.
If a file with the same name already exists in the repo, it will be skipped.
"""

import os
import zipfile
from pathlib import Path
from huggingface_hub import HfApi, list_repo_files
from tqdm import tqdm

import argparse

def compress_folder(folder_path, output_zip_path):
    """
    Compress only .pth, .txt, and .json files in a folder into a zip file.

    Args:
        folder_path: Path to the folder to compress
        output_zip_path: Path where the zip file will be created
    """
    print(f"Compressing {folder_path} to {output_zip_path} (only .pth, .txt, .json files)...")

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        folder_path = Path(folder_path)

        # Only include .pth, .txt, and .json files
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in {'.pth', '.txt', '.json'}:
                # Use relative path inside the zip file
                arcname = file_path.relative_to(folder_path.parent)
                zipf.write(file_path, arcname)

    print(f"Successfully compressed to {output_zip_path}")
    


def get_existing_files(repo_id, token=None):
    """
    Get list of existing files in the HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/repo_name")
        token: HuggingFace API token (optional)
    
    Returns:
        Set of existing filenames in the repository
    """
    try:
        files = list_repo_files(repo_id, token=token)
        return set(files)
    except Exception as e:
        print(f"Warning: Could not list existing files in repo: {e}")
        return set()


def upload_to_huggingface(zip_path, repo_id, token=None):
    """
    Upload a zip file to HuggingFace repository.
    
    Args:
        zip_path: Path to the zip file to upload
        repo_id: HuggingFace repository ID (e.g., "username/repo_name")
        token: HuggingFace API token (optional, will use cached token if not provided)
    """
    api = HfApi()
    
    filename = os.path.basename(zip_path)
    
    print(f"Uploading {filename} to {repo_id}...")
    
    try:
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"Successfully uploaded {filename}")
        return True
    except Exception as e:
        print(f"Error uploading {filename}: {e}")
        return False


def main(args):
    """
    Main function to compress folders and upload to HuggingFace.
    """
    # Configuration
    base_dir = args.base_dir
    repo_id = args.repo_id
    token = args.token
    
    # Create temporary directory for zip files
    temp_zip_dir = "temp_zips"
    os.makedirs(temp_zip_dir, exist_ok=True)
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return
    
    # Get existing files in the repository
    print(f"Checking existing files in repository {repo_id}...")
    existing_files = get_existing_files(repo_id, token)
    
    # Get all subdirectories in the base directory
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {base_dir}")
        return
    
    print(f"Found {len(subdirs)} folders to process")
    
    # Process each subdirectory
    for subdir in tqdm(subdirs, desc="Processing folders"):
        folder_path = os.path.join(base_dir, subdir)
        zip_filename = f"{subdir}.zip"
        zip_path = os.path.join(temp_zip_dir, zip_filename)
        
        # Check if file already exists in repo
        if zip_filename in existing_files:
            print(f"Skipping {zip_filename} - already exists in repository")
            continue
        
        # Compress the folder
        try:
            compress_folder(folder_path, zip_path)
        except Exception as e:
            print(f"Error compressing {folder_path}: {e}")
            continue
        
        # Upload to HuggingFace
        upload_success = upload_to_huggingface(zip_path, repo_id, token)
        
        # Clean up the zip file after upload
        if upload_success and os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Cleaned up temporary file {zip_path}")
    
    # Remove temporary directory if empty
    if os.path.exists(temp_zip_dir) and not os.listdir(temp_zip_dir):
        os.rmdir(temp_zip_dir)
        print(f"Removed temporary directory {temp_zip_dir}")
    
    print("\nAll done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="work_dirs/tokenizer_training")
    parser.add_argument("--repo_id", type=str, default="QingyuShi/SemanticTok")
    parser.add_argument("--token", type=str, default="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")
    args = parser.parse_args()
    base_dir = args.base_dir
    repo_id = args.repo_id
    token = args.token
    main(args)