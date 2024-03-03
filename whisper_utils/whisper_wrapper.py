import keyboard
import pyaudio
import wave
import threading
from openai import OpenAI
import whisper
import gc
import torch
from openai import OpenAI
import os

class KeyControlledRecorder:
    def __init__(self,whisper_type, key='t', output_filename='temp.wav',use_openai_whisper = False):

        self.whisper_type = whisper_type
        self.key = key
        self.output_filename = output_filename
        self.is_recording = False
        self.frames = []
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording_thread = None
        self.use_openai_whisper = use_openai_whisper

        if self.use_openai_whisper: # if use OpenAI Whisper
            print("-------------------OpenAI Whisper-------------------")
            self.client = OpenAI()
            self.api_key = os.environ["OPENAI_KEY"]
            self.client.api_key = self.api_key
        else:
            print("loading models...")
            self.whisper_model = whisper.load_model(self.whisper_type)


    def start_recording(self):
        if self.is_recording:
            return
        print("Recording...")
        self.is_recording = True
        self.frames = []

        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=44100,
                                  input=True,
                                  frames_per_buffer=1024)
        def record():
            while self.is_recording:
                data = self.stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)

        self.recording_thread = threading.Thread(target=record)
        self.recording_thread.start()

    def stop_recording(self):
        if not self.is_recording:
            return
        print("Stopped recording.")
        self.is_recording = False
        self.recording_thread.join()

        # ストリームを停止・閉じる
        self.stream.stop_stream()
        self.stream.close()

        # WAVファイルに書き込む
        wf = wave.open(self.output_filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"File saved: {self.output_filename}")

    def on_press(self, event):
        if event.name == self.key:
            self.start_recording()

    def on_release(self, event):
        if event.name == self.key:
            self.stop_recording()

    def convert_to_text(self):

        print("Processing Voice to Text")

        if self.use_openai_whisper: # if use OpenAI Whisper
            audio_file = open(self.output_filename, "rb")
            transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                    )
            audio_file.close()
            result_text = transcription.text
        
        else:   # if use whisper model on local
            result = self.whisper_model.transcribe(self.output_filename)
            result_text = result['text']

            del self.whisper_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



        return   result_text

