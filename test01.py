import torch
from faster_whisper import WhisperModel
from zhconv import convert

device = "cuda"
model_path = r"/media/martin/DATA/_ML/RemoteProject/FinetuneWhisper-Server/Model/checkpoint-100-2024Y_11M_21D_16h_50m_33s"
model = WhisperModel(model_path, device=device)

media_path = r"/media/martin/DATA/_ML/RemoteProject/FinetuneWhisper-Server/pre_data/preData_2024Y_11M_22D_14h_44m_02s/0_2024Y_11M_22D_14h_44m_02s.mp4"
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
                          no_speech_threshold=0.2,
                          max_initial_timestamp=9999999.0)

segments, info = result
book = ''
for segment in segments:
    text = segment.text.strip()
    text = convert(text, 'zh-cn')
    paragraph = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, text)
    print(paragraph)
