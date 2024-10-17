import os
import json
from zipfile import ZipFile
import subprocess
import ffmpeg
from glob import glob
import requests

import time


# 读取配置文件
def read_config(config_file_path):
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    return config


# 保存配置文件
def save_config(config_file_path, config):
    with open(config_file_path, 'w') as f:
        json.dump(config, f)


# 压缩文件夹的函数
def zip_folder(folder_path, zip_filename):
    with ZipFile(zip_filename, 'w') as zip:
        for folderName, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                filePath = os.path.join(folderName, filename)
                zip.write(filePath)


# 解压缩文件的函数
def unzip_file(zip_file, dest_folder):
    with ZipFile(zip_file, 'r') as zip:
        zip.extractall(dest_folder)


def download_m3u8_to_mp4(m3u8_url, output_path, temp_path):
    if os.path.exists(temp_path):
        os.remove(temp_path)

    command = [
        "ffmpeg",
        "-i", m3u8_url,
        "-c", "copy",
        temp_path
    ]

    subprocess.run(command, check=True)

    os.rename(temp_path, output_path)
    print('m3u8保存为{}'.format(output_path))


# 下载m3u8文件 https://upyun.luckly-mjw.cn/Assets/media-source/example/media/index.m3u8
def download_m3u8_list(m3u8_url_list, save_folder, temp_folder):
    temp_path = os.path.join(temp_folder, 'temp_m3u8.mp4')

    for i, url in enumerate(m3u8_url_list.split('\n')):
        print(i, url)
        if url.strip():
            loc_time = time.strftime('%YY_%mM_%dD_%Hh_%Mm_%Ss')
            filename = f"{i}_{loc_time}.mp4"
            save_path = os.path.join(save_folder, filename)

            print("下载: ", url.strip())
            download_m3u8_to_mp4(url.strip(), save_path, temp_path)
