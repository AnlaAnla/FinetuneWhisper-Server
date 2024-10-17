from utils.MergeLora import MergeLora

# MergeLora(lora_model_path=r"/media/martin/DATA/_ML/RemoteProject/FinetuneWhisper-Server/train_result/train/checkpoint-100-2024Y_10M_08D_17h_43m_12s/adapter_model",
#           model_save_dir=r"/media/martin/DATA/_ML/RemoteProject/FinetuneWhisper-Server/Model")
import sys
import os
if sys.prefix == "/media/martin/DATA/miniconda3/envs/yolov8":
    os.environ["LD_LIBRARY_PATH"] = "/media/martin/DATA/miniconda3/envs/yolov8/lib/python3.8/site-packages/nvidia/cudnn/lib"
aa = os.environ.get("LD_LIBRARY_PATH")
print(aa)
# comand = "ct2-transformers-converter --model Model/whisper-medium-finetune --output_dir Model/whisper-medium-finetune-ct2 --copy_files tokenizer.json preprocessor_config.json"
#
# ddd = os.system(comand)
# print(ddd)
