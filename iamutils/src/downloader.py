# downloader.py
import numpy as np
import cv2
from urllib.request import urlopen
from PIL import Image
import os
from .multi_processing import pool_worker


def download_image_by_url(urls, num_worker, save_folder):
    ''' Download images then save them all in save_folder
        Parameters:
            urls: list of image urls
            num_worker: A number of processor to use
            save_folder: A folder to use to save images
    '''
    inputs = [(url, sfolder) for (url, sfolder) in zip(urls, [save_folder]*len(urls))]
    ret = pool_worker(inputs=inputs, target=get_image_from_url, num_worker=num_worker)
    return ret


def get_image_from_url(url, save_folder):
    ''' Download an image then save it as an jpg format
    '''
    img = np.array(Image.open(urlopen(url)))    
    if save_folder != None:
        img = img[:, :, ::-1]
        cv2.imwrite(os.path.join(save_folder, url.split('/')[-1]) + '.jpg', 
                   img)
    return img


