import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import os
from datetime import datetime


SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_NAME = "base"

# "en" = English
# "bg" = Bulgarian
# None = Automatic detection

LANGUAGE = "en"

AUDIO_FILE = "recording.wav"
OUTPUT_FILE = "transcription.txt"


print("Loading Whisper AI model...")
model = whisper.load_model(MODEL_NAME)

print("Whisper is ready!")



def record_audio():

    print("\n SPEECH TO TEXT \n")
    duration = int(input("Recording duration in seconds: "))

    print("\nRecording starts in 2 seconds...")
    sd.sleep(2000)

    print("Listening... SPEAK NOW!")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )

    sd.wait()

    print("Recording finished!")

    write(AUDIO_FILE, SAMPLE_RATE, audio)

    print("Audio saved:", AUDIO_FILE)



def transcribe_audio():

    print("\nTranscribing audio... Please wait.")

    options = {}

    if LANGUAGE is not None:
        options["language"] = LANGUAGE

    result = model.transcribe(
        AUDIO_FILE,
        fp16=False,
        **options
    )

    text = result["text"].strip()

    print("\n YOUR TEXT \n")

    print(text)

    return text


def save_text(text):

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:

        file.write("\n\n")
        file.write("Date: " + str(datetime.now()))
        file.write("\n")
        file.write(text)

    print("\nText saved to:", OUTPUT_FILE)


def main():

    while True:

        print("\n1. Record speech")
        print("2. Exit")

        choice = input("\nChoose an option: ")

        if choice == "1":

            try:
                record_audio()

                text = transcribe_audio()

                save_text(text)

            except Exception as error:
                print("\nError:", error)

        elif choice == "2":

            print("Goodbye!")
            break

        else:

            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
