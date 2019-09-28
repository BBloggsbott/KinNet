import zipfile
import requests
import os
from pathlib import Path
import logging
import sys
import random
from fastai.vision import open_image
import matplotlib.pyplot as plt
import math

root = logging.getLogger()
root.setLevel(logging.INFO)

def get_title(filename):
    title_dict = {'m':'Mother', 'f':'Father', 's':'Son', 'd':'Daughter'}
    filename = str(filename).split('/')[-1]
    n = filename.split('_')[1]
    pc = filename.split('_')[-1][0]
    return "{}: {}".format(title_dict[filename[0]] if pc == '1' else title_dict[filename[1]], n)

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
    dirs = os.listdir(save_dir/'images')
    for i in dirs:
        try:
            (save_dir/'images'/i/'Thumbs.db').unlink()
        except:
            pass
    logging.info("Dataset can be found at {}".format(save_dir))
    return save_dir

class KinNetDataset:
    def __init__(self, n, d, bs):
        logging.info("Creating Dataset")
        self.data_dir = get_dataset(n, d)
        self.data_dir = self.data_dir / "images"
        self.parent_list = ["father", "mother"]
        self.child_list = ["son", "dau"]
        self.is_parent = lambda x: x.split(".")[0][-1] == "1"
        self.bs = bs
        
    def get_pair(self, file_name):
        file_name = file_name.split(".")
        n = int(file_name[0][-1])
        n = (n % 2) + 1
        file_name[0] = file_name[0][:-1] + str(n)
        return ".".join(file_name), True if n == 1 else False

    def get_random_kinship_pair(self):
        p = random.choice(self.parent_list)
        c = random.choice(self.child_list)
        return self.data_dir / (p+"-"+c)

    def get_quadruple_filenames(self):
        dir = self.get_random_kinship_pair()
        xp = random.choice(os.listdir(dir))
        xc, is_parent = self.get_pair(xp)
        if is_parent:
            xp, xc = xc, xp
        xp = dir / xp
        xc = dir / xc
        dir = self.get_random_kinship_pair()
        x_p_cap = random.choice(os.listdir(dir))
        if not self.is_parent(x_p_cap):
            x_p_cap, _ = self.get_pair(x_p_cap)
        x_p_cap = dir / x_p_cap
        dir = self.get_random_kinship_pair()
        x_c_cap = random.choice(os.listdir(dir))
        if self.is_parent(x_c_cap):
            x_c_cap, _ = self.get_pair(x_c_cap)
        x_c_cap = dir / x_c_cap
        return xp, xc, x_p_cap, x_c_cap

    def get_quadruple(self):
        files = self.get_quadruple_filenames()
        return [open_image(i) for i in files]

    def get_batch(self, size=10):
        return [self.get_quadruple() for _ in range(size)]

    def show_batch(self):
        n_images = self.bs*2
        plot_size = 4, math.ceil(n_images/4)
        self.fig, ax = plt.subplots(plot_size[1], plot_size[0])
        cur_row = 0
        cur_col = 0
        for i in range(self.bs):
            quad = self.get_quadruple_filenames()
            img1 = plt.imread(quad[0])
            img2 = plt.imread(quad[1])
            ax[cur_row, cur_col].axis('off')
            ax[cur_row, cur_col].imshow(img1)
            ax[cur_row, cur_col].set_title(get_title(quad[0]))
            cur_col += 1
            ax[cur_row, cur_col].axis('off')
            ax[cur_row, cur_col].imshow(img2)
            ax[cur_row, cur_col].set_title(get_title(quad[1]))
            cur_col += 1
            
            if cur_col >= plot_size[0]:
                cur_row += 1
                cur_col = 0
        for i in range(cur_col, 4):
            ax[cur_row, i].axis('off')
        plt.show()
        
    

