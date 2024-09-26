import gradio as gr
import os
import shutil
import datetime
import json
import time
from utils.utils import zip_folder, unzip_file, read_config, save_config, download_m3u8
from utils.VideoSplitter import VideoSplitter
from utils.FinetuneWhisper import FinetuneWhisper

# 保存配置文件的路径
Config_file_path = 'config.json'

# 服务器文件夹路径
Pre_data_path = 'pre_data'
Dataset_path = 'train_dataset'


# Gradio函数
def upload_media2server(media_files, urls, folder_name):
    if folder_name is None:
        return "请确认数据集名称"

    folder_path = os.path.join(Pre_data_path, folder_name)
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

    # # 保存配置文件
    # if config is not None:
    #     save_config(config)

    return "上传结束\n111"


def the1_media_split(t2_folder_name: str):
    video_splitter = VideoSplitter()
    media_folder = os.path.join(Pre_data_path, t2_folder_name)
    data_save_dir = os.path.join(Dataset_path, t2_folder_name)

    print("开始处理")
    video_splitter.run(media_folder, data_save_dir)
    print('处理结束')
    return f"处理结束"


def the5_finetune_whisper(_folder_name):
    train_data_path = os.path.join(Dataset_path, _folder_name)
    train_data_abspath = os.path.abspath(train_data_path)
    print('开始微调')
    FinetuneWhisper(train_data_abspath)
    return "微调结束"


def create_gradio_page():
    with gr.Blocks() as page:
        with gr.Tab("1-上传原始数据"):
            with gr.Row():
                media_files = gr.File(label="上传媒体数据", file_count="multiple")
                urls = gr.Textbox(label="输入m3u8链接, 多个链接每行一个")

            with gr.Row():
                # t1_folder_name = "data_" + time.strftime('%Y-%m-%d-%H-%M-%S')
                t1_folder_name = gr.Dropdown(choices=["folder1", "folder2", "folder3"], label="选择数据集名称")
            with gr.Row():
                t1_result = gr.Textbox(label="上传结果")

            t1_upload_bth = gr.Button(value="上传数据", variant='primary')
            t1_upload_bth.click(fn=upload_media2server, inputs=[media_files, urls, t1_folder_name], outputs=[t1_result])

        with gr.Tab("2-处理数据"):
            with gr.Row():
                t2_folder_name = gr.Dropdown(choices=os.listdir(Pre_data_path), label="选择数据集名称")
            with gr.Row():
                t2_result = gr.Textbox(label="处理结果")

            t2_btn = gr.Button(value="开始处理", variant='primary')
            t2_btn.click(fn=the1_media_split, inputs=[t2_folder_name], outputs=[t2_result])

        with gr.Tab("3-上传Label-Studio和Minio"):
            config = gr.JSON(value=read_config(Config_file_path), label="配置信息")
            with gr.Row():
                t3_folder_name = gr.Dropdown(choices=os.listdir(Dataset_path), label="选择数据集名称")

            t3_upload_btn = gr.Button(value="上传Label-Studio和Minio", variant='primary')

        with gr.Tab("4-下载训练数据"):
            zip_file = gr.File(label="上传Zip文件")

        with gr.Tab("5-微调模型"):
            with gr.Row():
                t5_folder_name = gr.Dropdown(choices=os.listdir(Dataset_path), label="选择数据集名称")
            with gr.Row():
                t5_result = gr.Textbox(label="微调结果")
            t5_train_btn = gr.Button(value="开始微调", variant='primary')
            t5_train_btn.click(fn=the5_finetune_whisper, inputs=[t5_folder_name], outputs=[t5_result])

        with gr.Tab("6-试用模型"):
            with gr.Row():
                t6_btn = gr.Button(value="语音识别", variant='primary')

    page.launch(server_name='0.0.0.0', server_port=1234)


if __name__ == "__main__":
    create_gradio_page()
