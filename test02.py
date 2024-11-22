import os
import subprocess
import time


def download_m3u8_to_mp4(m3u8_url, output_path, temp_path):

    command = [
        "ffmpeg",
        "-i", m3u8_url,
        "-c", "copy",
        temp_path
    ]

    subprocess.run(command, check=True)

    os.rename(temp_path, output_path)
    print('m3u8保存为{}'.format(output_path))

m3u8_url = "https://upyun.luckly-mjw.cn/Assets/media-source/example/media/index.m3u8"

download_m3u8_to_mp4(m3u8_url, 'temp/test.mp4', 'temp/test.mp4')