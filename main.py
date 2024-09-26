import gradio as gr
import os
import shutil
import datetime
import json
import time
from utils.utils import zip_folder, unzip_file, read_config, save_config, download_m3u8
from utils.VideoSplitter import VideoSplitter

# 保存配置文件的路径
config_file_path = 'config.json'

# 服务器文件夹路径
pre_data_path = 'pre_data'
dataset_path = 'train_dataset'


# Gradio函数
def gradio_interface(media_files, urls, folder_name):
    if folder_name is None:
        return "请确认数据集名称"

    folder_path = os.path.join(pre_data_path, folder_name)
    # 处理上传的媒体文件
    if media_files is not None:

        os.makedirs(folder_path, exist_ok=True)
        for media_file in media_files:
            print("保存: ", media_file, folder_path)
            shutil.copy(media_file, folder_path)

    # 处理m3u8链接
    if urls is not None:
        for url in urls.split('\n'):
            print(url)
            if url.strip():
                download_m3u8(url.strip(), save_folder=folder_path)
                print("下载: ", url.strip())

    # # 压缩文件夹并下载
    # if config['zip_folder']:
    #     zip_filename = 'server_folder.zip'
    #     zip_folder(pre_data_path, zip_filename)
    #     with open(zip_filename, 'rb') as f:
    #         zip_data = f.read()
    #     os.remove(zip_filename)
    #     return zip_data
    #
    # # 保存配置文件
    # if config is not None:
    #     save_config(config)
    #
    # # 解压缩文件
    # if zip_file is not None:
    #     unzip_file(zip_file.name, pre_data_path)

    return "上传结束\n111"


def the1_media_split(t2_folder_name: str):
    video_splitter = VideoSplitter()
    media_folder = os.path.join(pre_data_path, t2_folder_name)
    data_save_dir = os.path.join(dataset_path, t2_folder_name)

    print("开始处理")
    video_splitter.run(media_folder, data_save_dir)
    print('处理结束')
    return "处理结束 " + time.strftime('%Y-%m-%d-%H-%M-%S')


def create_gradio_page():
    with gr.Blocks() as page:
        with gr.Tab("上传原始数据"):
            with gr.Row():
                media_files = gr.File(label="上传媒体数据", file_count="multiple")
                urls = gr.Textbox(label="输入m3u8链接, 多个链接每行一个")

            with gr.Row():
                # t1_folder_name = "data_" + time.strftime('%Y-%m-%d-%H-%M-%S')
                t1_folder_name = gr.Dropdown(choices=["folder1", "folder2", "folder3"], label="选择数据集名称")
            with gr.Row():
                t1_result = gr.Textbox(label="上传结果")

            t1_upload_bth = gr.Button(value="上传数据", variant='primary')
            t1_upload_bth.click(fn=gradio_interface, inputs=[media_files, urls, t1_folder_name], outputs=[t1_result])

        with gr.Tab("处理数据"):
            with gr.Row():
                t2_folder_name = gr.Dropdown(choices=os.listdir(pre_data_path), label="选择数据集名称")
            with gr.Row():
                t2_result = gr.Textbox(label="处理结果")

            t2_btn = gr.Button(value="开始处理", variant='primary')
            t2_btn.click(fn=the1_media_split, inputs=[t2_folder_name], outputs=[t2_result])

        with gr.Tab("上传Label-Studio和Minio"):
            config = gr.JSON(value=read_config(config_file_path), label="配置信息")
            with gr.Row():
                t3_folder_name = gr.Dropdown(choices=os.listdir(dataset_path), label="选择数据集名称")

        with gr.Tab("下载训练数据"):
            zip_file = gr.File(label="上传Zip文件")

    page.launch(server_port=1234)


if __name__ == "__main__":
    create_gradio_page()
