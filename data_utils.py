import zipfile
import requests
import os
from pathlib import Path
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def check_downloads_directory(dir_name = "downloads", dataset = 1):
    dir_name = Path(dir_name)
    if not os.path.isdir(dir_name):
        logging.info("'{}' directory does not exist. Creating it.".format(dir_name))
        os.makedirs(dir_name)
        return False
    if dataset == 1:
        if not os.path.isfile(dir_name / "KinFaceW-I.zip"):
            return False
        else:
            return True
    elif dataset == 2:
        if not os.path.isfile(dir_name / "KinFaceW-II.zip"):
            return False
        else:
            return True


def download_dataset(dir_name = "data", dataset = 1):
    if check_downloads_directory(dir_name, dataset):
        logging.info("Dataset exists. Skipping Download")
    else:
        logging.info("Dataset does not exist. Downloading...")
        download_link = "http://www.kinfacew.com/dataset/KinFaceW-I.zip" if dataset == 1 else "http://www.kinfacew.com/dataset/KinFaceW-II.zip"
        dir_name = Path(dir_name)
        response = requests.get(download_link)
        file_name = download_link.split("/")[-1]
        with open(dir_name / file_name, 'wb+') as downloaded_file:
            downloaded_file.write(response.content)
            downloaded_file.close()
        logging.info("Dataset Downloaded")

def unzip_data(dataset = 1, data_dir = "data"):
    if check_downloads_directory(data_dir, dataset):
        zipfile_name = "KinFaceW-I.zip" if dataset == 1 else "KinFaceW-II.zip"
        data_dir = Path(data_dir)
        logging.info("Starting Unzip of {}".format(zipfile_name))
        zip_ref = zipfile.ZipFile(data_dir / zipfile_name, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        logging.info("Dataset Unzipped")
        return data_dir / zipfile_name.split(".")[0]

def get_dataset(dataset = 1, dir_name = "data"):
    download_dataset(dir_name, dataset)
    save_dir = unzip_data(dataset, dir_name)
    logging.info("Dataset can be found at {}".format(save_dir))
    return save_dir
