import torch
from transformers import WhisperForConditionalGeneration, AutoProcessor, pipeline, WhisperFeatureExtractor, \
    WhisperTokenizerFast, WhisperProcessor

import os

from peft import LoraConfig, PeftModel, PeftConfig
import evaluate
import datetime
import srt
from zhconv import convert


class RecognizeAudio:
    def __init__(self, lora_model_path):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0"

        # self.lora_model_path = lora_model_path
        local_files_only = False

        peft_config = PeftConfig.from_pretrained(lora_model_path)
        # 获取Whisper的基本模型
        base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path,
                                                                     device_map=device,
                                                                     local_files_only=local_files_only)
        # 与Lora模型合并
        model = PeftModel.from_pretrained(base_model, lora_model_path, local_files_only=local_files_only)
        model.to(device)
        model.config.use_cache = True

        feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path,
                                                                    local_files_only=local_files_only)
        tokenizer = WhisperTokenizerFast.from_pretrained(peft_config.base_model_name_or_path,
                                                         local_files_only=local_files_only)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            torch_dtype=torch_dtype,
            # device=device,
        )

    def run(self, media_path):
        result = self.pipe(media_path, return_timestamps=True)
        # 创建SRT字幕文件
        subtitles = []
        temp_end_time = 0
        chunks_start_time = 0
        for chunk in result['chunks']:
            # 每30s为一个周期, 进行正确的时间累加
            if temp_end_time > chunk['timestamp'][0]:
                chunks_start_time += 30
                print('')
            temp_end_time = chunk['timestamp'][1]

            start_time = chunk['timestamp'][0] + chunks_start_time
            end_time = chunk['timestamp'][1] + chunks_start_time
            text = convert(chunk['text'], 'zh-cn')

            paragraph = ("[%.2fs -> %.2fs] %s" % (start_time, end_time, text))
            subtitles.append(paragraph)

        book = '\n'.join(subtitles)
        return book
