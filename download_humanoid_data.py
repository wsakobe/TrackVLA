import os
import requests
import zipfile
from tqdm import tqdm
import shutil

def download_file(url, filename):
    print(f"Downloading from {url} ...")
    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            raise Exception(f"Download failed with status code: {response.status_code}")

        total_size = int(response.headers.get('content-length', 0))
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Downloaded to {filename}")

def unzip_folder(zip_path):
    print(f"Extracting {zip_path} in current directory...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

def download_and_extract_zip(url, zip_filename='temp_download.zip'):
    download_file(url, zip_filename)
    unzip_folder(zip_filename)
    os.remove(zip_filename)
    print(f"Removed temporary file: {zip_filename}")

if __name__ == '__main__':
    url = 'https://0x0.st/80mQ.zip'
    download_and_extract_zip(url)