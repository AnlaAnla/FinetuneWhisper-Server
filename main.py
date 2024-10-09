import gradio as gr
import sys
import os
import shutil
import datetime
import json
import time
from utils.utils import zip_folder, unzip_file, read_config, save_config, download_m3u8
from utils.VideoSplitter import VideoSplitter
from utils.FinetuneWhisper import FinetuneWhisper
from utils.Upload2DataServer import Upload2DataServer
from utils.ExportData import ExportData
from utils.RecognizeAudio import RecognizeAudio

if sys.prefix == "/media/martin/DATA/miniconda3/envs/yolov8":
    os.environ["LD_LIBRARY_PATH"] = "/media/martin/DATA/miniconda3/envs/yolov8/lib/python3.8/site-packages/nvidia/cudnn/lib"


# 保存配置文件的路径
Config_file_path = 'config.json'
Config = read_config(Config_file_path)

# 服务器文件夹路径
Pre_data_path = 'pre_data'
Dataset_path = 'train_dataset'
Train_result_path = 'train_result/train'
Temp_path = 'temp'
Model_path = 'Model'

os.makedirs(Pre_data_path, exist_ok=True)
os.makedirs(Dataset_path, exist_ok=True)
os.makedirs(Temp_path, exist_ok=True)


# Gradio函数
def the1_upload_media2server(media_files, urls, folder_name):
    if folder_name is None:
        folder_name = "preData_" + time.strftime('%YY_%mM_%dD_%Hh_%Mm_%Ss')

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

    return f"上传结束 - {time.strftime('%YY_%mM_%dD_%Hh_%Mm_%Ss')}"


def the2_split_upload(_folder_name: str, _project_name: str):
    if _project_name is None:
        return "请输入label-studio项目名称"

    video_splitter = VideoSplitter()
    media_folder = os.path.join(Pre_data_path, _folder_name)
    data_save_dir = os.path.join(Temp_path, _folder_name)

    print("开始处理")
    video_splitter.run(media_folder, data_save_dir)
    print('处理结束')
    minio_access_key = Config['minio_access_key']
    minio_secret_key = Config['minio_secret_key']
    label_studio_token = Config['label_studio_token']
    remote_data_server_ip = Config['remote_data_server_ip']
    label_studio_url = Config['label_studio_url']

    Upload2DataServer(project_name=_project_name,
                      audio_data_dir_path=data_save_dir,
                      minio_access_key=minio_access_key,
                      minio_secret_key=minio_secret_key,
                      label_studio_token=label_studio_token,
                      remote_data_server_ip=remote_data_server_ip,
                      label_studio_url=label_studio_url)
    shutil.rmtree(data_save_dir)

    return f"处理和上传结束 - {time.strftime('%YY_%mM_%dD_%Hh_%Mm_%Ss')}"


def the3_download_data(_project_id, _folder_name: str):
    if int(_project_id) == 0:
        return "请输入label-studio项目ID"

    if _folder_name is None or _folder_name == '':
        _folder_name = f"DataSet_{time.strftime('%YY_%mM_%dD_%Hh_%Mm_%Ss')}"

    audio_data_dir_path = os.path.abspath(os.path.join(Dataset_path, _folder_name))
    os.makedirs(audio_data_dir_path, exist_ok=True)

    label_studio_token = Config['label_studio_token']
    label_studio_url = Config['label_studio_url']

    ExportData(project_id=_project_id,
               audio_data_dir_path=audio_data_dir_path,
               label_studio_url=label_studio_url,
               label_studio_token=label_studio_token)

    return f"下载结束, 保存为: {_folder_name}"


def the4_finetune_whisper(_folder_name):
    train_data_path = os.path.join(Dataset_path, _folder_name)
    train_data_abspath = os.path.abspath(train_data_path)
    print('开始微调')
    FinetuneWhisper(train_data_abspath)
    return f"微调结束 - {time.strftime('%YY_%mM_%dD_%Hh_%Mm_%Ss')}"


def the5_recognize_audio(_model_folder_name, _media_path):
    if _media_path is None:
        return "请放入音频或视频"

    _model_folder_path = os.path.join(Train_result_path, _model_folder_name, "adapter_model")
    recognizer = RecognizeAudio(_model_folder_path)
    book = recognizer.run(_media_path)
    print(book)
    return book



def refresh_list():
    folder_name1 = gr.Dropdown(choices=os.listdir(Pre_data_path), label="选择原始数据")
    folder_name2 = gr.Dropdown(choices=os.listdir(Dataset_path), label="选择数据集名称")
    model_folder_name = gr.Dropdown(choices=os.listdir(Train_result_path), label="选择模型名称")
    return folder_name1, folder_name1, folder_name2, model_folder_name


def create_gradio_page():
    with gr.Blocks() as page:
        with gr.Tab("1-上传原始数据"):
            with gr.Row():
                media_files = gr.File(label="上传媒体数据", file_count="multiple")
                urls = gr.Textbox(label="输入m3u8链接, 多个链接每行一个")

            with gr.Row():
                gr.Label("选择上传的位置, 如果不选, 则自动创建")
                t1_folder_name = gr.Dropdown(choices=os.listdir(Pre_data_path), label="选择原始数据")
            with gr.Row():
                t1_result = gr.Textbox(label="上传结果")

            t1_upload_bth = gr.Button(value="上传数据", variant='primary')
            t1_upload_bth.click(fn=the1_upload_media2server, inputs=[media_files, urls, t1_folder_name],
                                outputs=[t1_result])

        with gr.Tab("2-处理并上传Label-Studio和Minio"):
            with gr.Row():
                t2_folder_name = gr.Dropdown(choices=os.listdir(Pre_data_path), label="选择原始数据")
                t2_project_name = gr.Textbox(lines=1, label="输入label-studio项目名称, 注意不要和现有项目重名")
            with gr.Row():
                t2_result = gr.Textbox(label="处理结果")

            t2_btn = gr.Button(value="开始处理", variant='primary')
            t2_btn.click(fn=the2_split_upload, inputs=[t2_folder_name, t2_project_name], outputs=[t2_result])

        with gr.Tab("3-下载训练数据"):
            with gr.Row():
                t3_project_id = gr.Number(label="输入label-studio项目ID")
                t3_folder_name = gr.Textbox(lines=1, label="输入保存的数据集名称, 不输入则自动创建")
            with gr.Row():
                t3_result = gr.Textbox(label="处理结果")
            t3_download_btn = gr.Button(value="开始下载", variant='primary')
            t3_download_btn.click(fn=the3_download_data,
                                  inputs=[t3_project_id, t3_folder_name],
                                  outputs=[t3_result])

        with gr.Tab("4-微调模型"):
            with gr.Row():
                t4_folder_name = gr.Dropdown(choices=os.listdir(Dataset_path), label="选择数据集名称")
            with gr.Row():
                t4_result = gr.Textbox(label="微调结果")
            t4_train_btn = gr.Button(value="开始微调", variant='primary')
            t4_train_btn.click(fn=the4_finetune_whisper, inputs=[t4_folder_name], outputs=[t4_result])

        with gr.Tab("5-试用模型"):
            with gr.Row("选择模型"):
                t5_select_model_name = gr.Text(f"{os.listdir(Model_path)}")
                t5_model_folder_name = gr.Dropdown(choices=os.listdir(Train_result_path), label="选择模型名称")
            t5_change_model_btn = gr.Button(value="切换模型", variant='primary')
            with gr.Row("进行识别"):
                t5_audio_file = gr.Audio(sources="upload", type="filepath")
                t5_result = gr.Textbox()
            t5_btn = gr.Button(value="语音识别", variant='primary')
            t5_btn.click(fn=the5_recognize_audio, inputs=[t5_model_folder_name, t5_audio_file], outputs=[t5_result])

        refresh_btn = gr.Button("🌀刷新")
        refresh_btn.click(fn=refresh_list, outputs=[t1_folder_name, t2_folder_name, t4_folder_name, t5_model_folder_name])

    page.launch(server_name='0.0.0.0', server_port=1234)


if __name__ == "__main__":
    create_gradio_page()
