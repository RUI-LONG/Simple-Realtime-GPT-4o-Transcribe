import io
import queue
import asyncio
from openai import OpenAI
import speech_recognition as sr

client = OpenAI(
    api_key="YOUR API KEY HERE",
)


MIC_INDEX = 5
# TODO: Use this to get your mic index
# available_mics = sr.Microphone.list_working_microphones()
# available_mics = list(set(available_mics))
# print(available_mics)


class VoiceTranscriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

        self.phrase_time_limit = 10
        self.audio_queue = queue.Queue()
        self.source = None

    def _record_callback(self, _, audio: sr.AudioData):
        """Callback to handle incoming audio asynchronously."""
        try:
            raw_audio = audio.get_raw_data()
            self.audio_queue.put(raw_audio)
        except Exception as e:
            print("[Record Error]:", e)

    async def _record_loop(self):
        """Starts background listening and keeps the event loop alive."""
        mic = sr.Microphone(device_index=MIC_INDEX)

        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.source = source
            print("Microphone calibrated.")

        self.recognizer.listen_in_background(
            mic, self._record_callback, phrase_time_limit=self.phrase_time_limit
        )
        print("Listening started.")

    def is_valid_logprobs(self, logprobs, threshold=-0.1) -> bool:
        if not logprobs:
            return False
        avg_logprob = sum(lp.logprob for lp in logprobs) / len(logprobs)
        return avg_logprob > threshold

    async def _process_audio(self):
        while True:
            try:
                raw_audio = self.audio_queue.get()
            except queue.Empty:
                continue

            try:
                audio_data = sr.AudioData(
                    raw_audio, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH
                )
                wav_data = audio_data.get_wav_data()
                wav_stream = io.BytesIO(wav_data)
                wav_stream.name = "_temp/audio.wav"
                wav_stream.seek(0)
                response = client.audio.transcriptions.create(
                    # model="gpt-4o-transcribe",
                    model="gpt-4o-mini-transcribe",
                    file=wav_stream,
                    temperature=0.2,
                    response_format="json",
                    include=["logprobs"],
                    timeout=5,
                )

                print(
                    f"Is valid transcript: {self.is_valid_logprobs(response.logprobs)}"
                )
                print(response.text)

            except Exception as e:
                print("[Transcription Error]:", e)

    async def run(self):
        """Run recording and transcription tasks concurrently."""
        await asyncio.gather(self._record_loop(), self._process_audio())


if __name__ == "__main__":

    transcriber = VoiceTranscriber()
    asyncio.run(transcriber.run())
