import os
import json
import pandas as pd
import numpy as np
import requests


class ExportData:
    def __init__(self, project_id,
                 audio_data_dir_path,
                 label_studio_url,
                 label_studio_token):
        self.label_studio_url = label_studio_url
        self.label_studio_token = label_studio_token
        self.project_id = project_id
        self.audio_data_dir_path = audio_data_dir_path

        if not os.path.exists(os.path.join(self.audio_data_dir_path, 'audio')):
            os.mkdir(os.path.join(self.audio_data_dir_path, 'audio'))

        json_datas = self.__get_label(project_id)

        metadata = []
        for json_data in json_datas:
            audio_path = 'audio/' + f"id{project_id}_" + os.path.split(json_data['audio'])[-1]

            try:
                self.__download_file(json_data['audio'], os.path.join(self.audio_data_dir_path, audio_path))
                metadata.append([audio_path, json_data['transcription']])
            except Exception as e:
                print("下载异常", e)

        meta_data = np.array(metadata)
        print(meta_data)
        meta_data_save_path = os.path.join(audio_data_dir_path, 'metadata.csv')

        # 判断并生成或合并csv label
        if not os.path.exists(meta_data_save_path):
            meta_data = pd.DataFrame(meta_data, columns=['file_name', 'sentence'])
            meta_data.to_csv(meta_data_save_path, encoding='utf-8', index=False)
        else:
            old_meta_data = pd.read_csv(meta_data_save_path, encoding='utf-8')
            old_meta_data = np.array(old_meta_data)
            merged_meta_data = np.concatenate((old_meta_data, meta_data), axis=0)
            merged_meta_data = pd.DataFrame(merged_meta_data, columns=['file_name', 'sentence'])

            # 清楚重复数据
            merged_meta_data = merged_meta_data.drop_duplicates(subset=None, keep='first', inplace=False)
            merged_meta_data.to_csv(meta_data_save_path, encoding='utf-8', index=False)

        print('下载结束')

    def __get_label(self, _project_id):
        url = f"{self.label_studio_url}/api/projects/{_project_id}/export?exportType=JSON_MIN"
        headers = {
            "Authorization": f"Token {self.label_studio_token}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers)
        return response.json()

    def __download_file(self, url, save_path):
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            # 打开一个文件用于写入
            with open(save_path, 'wb') as f:
                # 迭代读取数据
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # 过滤掉keep-alive的情况
                        f.write(chunk)
            print(f'文件下载成功, {save_path}: {url}')
        else:
            print('文件下载失败，错误代码:', response.status_code)
