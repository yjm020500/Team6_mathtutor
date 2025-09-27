import sounddevice as sd
import numpy as np
import wave
import whisper
from scipy.signal import resample_poly

# ===== 설정 =====
RATE_IN = 48000           # 실제 마이크 샘플레이트 (보통 48kHz)
RATE_OUT = 16000          # Whisper에 맞는 샘플레이트
CHANNELS = 1
OUTPUT_WAV = "recorded.wav"
OUTPUT_TXT = "transcript.txt"
#MODEL_NAME = "small"
MODEL_NAME = "medium"
GAIN = 3.0                # 녹음 볼륨 증폭 배율

recording = False
audio_data = []
stream = None

def start_recording():
    global recording, audio_data, stream
    print(" 녹음 시작! (다시 s + 엔터 누르면 중지)")
    audio_data = []
    recording = True

    def callback(indata, frames, time, status):
        if recording:
            audio = indata[:, 0].astype(np.float32)  # 1채널 기준
            audio = audio * GAIN
            audio = np.clip(audio, -32768, 32767).astype(np.int16)
            # 48kHz → 16kHz 리샘플링
            audio_16k = resample_poly(audio, up=1, down=3).astype(np.int16)
            audio_data.append(audio_16k)

    stream = sd.InputStream(samplerate=RATE_IN, channels=CHANNELS, dtype='int16', device=???, callback=callback)
    stream.start()

def stop_recording():
    global recording, stream
    recording = False
    stream.stop()
    stream.close()
    print(" 녹음 중지!")

    audio_np = np.concatenate(audio_data)
    with wave.open(OUTPUT_WAV, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE_OUT)
        wf.writeframes(audio_np.tobytes())
    print(f" 저장됨: {OUTPUT_WAV}")

    transcribe_with_whisper(OUTPUT_WAV, OUTPUT_TXT)

def transcribe_with_whisper(wav_path, txt_path):
    print(f" Whisper {MODEL_NAME} 로드 중…")
    model = whisper.load_model(MODEL_NAME)
    print(f" 음성 인식 중…")
    result = model.transcribe(wav_path, language="ko")

    text = result["text"]
    print(" 인식 결과:")
    print(text)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f" 텍스트 저장됨: {txt_path}")

if __name__ == "__main__":
    print(" 's' + 엔터 → 녹음 시작/중지, 'q' + 엔터 → 종료")
    while True:
        cmd = input("> ").strip()
        if cmd == "s":
            if not recording:
                start_recording()
            else:
                stop_recording()
        elif cmd == "q":
            if recording:
                stop_recording()
            print("종료합니다.")
            break
        else:
            print(" 's' 또는 'q'를 입력해주세요.")

