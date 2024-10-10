import os

from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast,\
    WhisperProcessor
from peft import PeftModel, PeftConfig


class MergeLora:
    def __init__(self, lora_model_path, model_save_dir):

        # lora_model_path = "reach-vb/train/checkpoint-100"
        # model_save_dir = "Model"
        self.lora_model_path = lora_model_path
        self.model_save_dir = model_save_dir

    def run(self):
        # # 获取Lora配置参数
        peft_config = PeftConfig.from_pretrained(self.lora_model_path)
        # 获取Whisper的基本模型
        base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map="cuda:0")
        # 与Lora模型合并
        model = PeftModel.from_pretrained(base_model, self.lora_model_path)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer = WhisperTokenizerFast.from_pretrained(peft_config.base_model_name_or_path)
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)

        # 合并参数
        model = model.merge_and_unload()
        model.train(False)

        # 保存的文件夹路径
        if peft_config.base_model_name_or_path.endswith("/"):
            peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
        save_directory = os.path.join(self.model_save_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
        os.makedirs(save_directory, exist_ok=True)

        # 保存模型到指定目录中
        model.save_pretrained(save_directory, max_shard_size='4GB')
        feature_extractor.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)
        print(f'合并模型保存在：{save_directory}')

        ct2_save_directory = os.path.join(self.model_save_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune-ct2')
        command = f"ct2-transformers-converter --model {save_directory} --output_dir {ct2_save_directory} --copy_files tokenizer.json preprocessor_config.json"

        print("执行ct2格式转化")
        retVal = os.system(command)
        print("ct2模型转化完成:{}".format(retVal))

        return ct2_save_directory
