import pandas as pd
from minio import Minio
import datetime
import numpy as np
import requests
import json
import os

audio_data_dir_path = "temp/Data_2024Y_09M_27D_11h_05m_29s"
audio_data_dir_path = os.path.abspath(audio_data_dir_path)

print('连接minio')
minio_client = Minio("truenas.lan:9000",
                     access_key="TQUDCSxXiw5D5hPBbm5J",
                     secret_key="DB63Vk3qU81LQ0szSM7DyrL9rfswJG2qmjJ0BAqU",
                     secure=False)
print('minio成功连接')
bucket_name = "label-studio-data"


data_dir_name = os.path.split(audio_data_dir_path)[-1]

print(f"开始向minio上传: {audio_data_dir_path}")
for root, dirs, files in os.walk(audio_data_dir_path):
    for file in files:
        # 构造源文件路径
        file_path = os.path.join(root, file)
        # 保留文件夹结构,构造在Minio存储桶中的对象名称
        object_name = os.path.relpath(file_path, audio_data_dir_path)
        object_name = object_name.replace('\\', '/')

        if data_dir_name:
            object_name = data_dir_name + '/' + object_name
        try:
            # 上传文件
            minio_client.fput_object(bucket_name, object_name, file_path)
            print(f'上传minio: {file_path} as {object_name}')
        except Exception as e:
            print('minio错误: ', e)