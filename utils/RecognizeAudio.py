import torch
from faster_whisper import WhisperModel
from zhconv import convert


class RecognizeAudio:
    def __init__(self, model_path):
        device = "cuda"

        self.model = WhisperModel(model_path, device=device)

    def run(self, media_path):
        result = self.model.transcribe(media_path, beam_size=5, word_timestamps=True,
                                       language='zh')

        segments, info = result
        book = ''
        for segment in segments:
            text = segment.text.strip()
            text = convert(text, 'zh-cn')
            paragraph = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, text)
            print(paragraph)

            book += f'{paragraph}\n'
            yield book
