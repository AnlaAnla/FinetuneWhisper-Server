from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from datasets import Audio
import torch
import time

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torchaudio
import os

from datasets import load_dataset, DatasetDict
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
import evaluate

from transformers import Seq2SeqTrainingArguments, BitsAndBytesConfig
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


class FinetuneWhisper:
    def __init__(self, dataset_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # model_name_or_path = "openai/whisper-large-v3"
        model_name_or_path = "/media/martin/DATA/_ML/Model/whisper-large-v3-turbo"
        task = "transcribe"

        # 此处加载我的数据集
        # dataset_path = r"D:\Code\ML\Audio\card_audio_data01\project-10-at-2024-09-23-09-07-87b7a955"
        language = "Chinese"
        print(f"默认语言为:{language}, 数据集路径为:{dataset_path}")
        # common_voice = DatasetDict()
        # common_voice["train"] = load_dataset("audiofolder", data_dir=dataset_path, split="train")
        # common_voice["test"] = load_dataset("audiofolder", data_dir=dataset_path, split="train")
        common_voice = load_dataset("audiofolder", data_dir=dataset_path, split="train").train_test_split(test_size=0.1)
        common_voice['train'] = load_dataset("audiofolder", data_dir=dataset_path, split="train")

        print(common_voice)
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
        print(common_voice["train"][0])

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

        common_voice = common_voice.map(self.prepare_dataset, remove_columns=common_voice.column_names["train"],
                                        num_proc=1)
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        # metric = evaluate.load("wer")
        model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                quantization_config=BitsAndBytesConfig(
                                                                    load_in_8bit=True),
                                                                device_map="cuda:0")
        model = prepare_model_for_kbit_training(model)

        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

        # 查看参与训练的参数
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        training_args = Seq2SeqTrainingArguments(
            output_dir="train_result/train",  # change to a repo name of your choice
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-4,
            warmup_steps=30,
            num_train_epochs=1,
            eval_strategy="steps",
            fp16=False,
            per_device_eval_batch_size=8,
            generation_max_length=128,
            logging_steps=50,
            max_steps=100,  # only for testing purposes, remove this from your final run :)
            remove_unused_columns=False,
            # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
            label_names=["labels"],  # same reason as above
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=common_voice["train"],
            eval_dataset=common_voice["test"],
            data_collator=data_collator,
            # compute_metrics=compute_metrics,
            tokenizer=self.processor.feature_extractor,
            callbacks=[SavePeftModelCallback],
        )
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

        trainer.train()

    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = \
            self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}-{time.strftime('%YY_%mM_%dD_%Hh_%Mm_%Ss')}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control
