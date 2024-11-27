import torch
from faster_whisper import WhisperModel
from zhconv import convert

device = "cuda"
# model_path = "medium"
model_path = r"/media/martin/DATA/_ML/RemoteProject/FinetuneWhisper-Server/Model/checkpoint-100-2024Y_11M_25D_17h_26m_29s"
model = WhisperModel(model_path, device=device)

media_path = r"pre_data/preData_2024Y_11M_25D_17h_45m_46s/tt184__0_2024Y_11M_14D_13h_01m_37s.mp3"
vad_param = {
    "threshold": 0.5,
    "min_speech_duration_ms": 1000,
    "min_silence_duration_ms": 100,
    "max_speech_duration_s": 30,
    "speech_pad_ms": 2000
}

result = model.transcribe(media_path, beam_size=5, language="zh",
                          vad_filter=True,
                          vad_parameters=vad_param,
                          no_speech_threshold=0.4,
                          max_initial_timestamp=9999999.0
                          )
print('+'*40)
segments, info = result
book = ''
for segment in segments:
    text = segment.text.strip()
    text = convert(text, 'zh-cn')
    paragraph = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, text)
    print(paragraph)
print('+'*40)
