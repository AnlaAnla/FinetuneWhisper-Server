import torch
from faster_whisper import WhisperModel
from zhconv import convert


class RecognizeAudio:
    def __init__(self, model_path):
        device = "cuda"

        self.model = WhisperModel(model_path, device=device)

    def run(self, media_path):
        vad_param = {
            "threshold": 0.5,
            "min_speech_duration_ms": 1000,
            "min_silence_duration_ms": 100,
            "max_speech_duration_s": 30,
            "speech_pad_ms": 2000
        }
        result = self.model.transcribe(media_path, beam_size=5, language="zh",
                          vad_filter=True,
                          vad_parameters=vad_param,
                          no_speech_threshold=0.2,
                          max_initial_timestamp=9999999.0)

        segments, info = result
        book = ''
        for segment in segments:
            text = segment.text.strip()
            text = convert(text, 'zh-cn')
            paragraph = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, text)
            print(paragraph)

            book += f'{paragraph}\n'
            yield book
