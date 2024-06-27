# Video GIF Extractor

This script processes a long video, transcribes its audio using OpenAI Whisper, and extracts suitable sentences to create GIFs with captions. The selection of suitable sentences is based on certain heuristics indicating their potential for being interesting or engaging as GIFs.

## Features

- **Transcription**: Uses OpenAI Whisper to transcribe the audio from the video.
- **Sentence Selection**: Applies heuristics to determine if a sentence is suitable for a GIF.
- **GIF Creation**: Extracts video segments corresponding to the selected sentences and creates GIFs with captions.

## Requirements

- Python 3.6+
- Required Python packages:
  - `whisper`
  - `moviepy`
  - `spacy`
  - `nltk`
  - `re`

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/SyedMa3/gif-maker.git
    cd gif-maker
    ```

2. **Install the required packages:**

    ```sh
    pip install whisper moviepy spacy nltk
    ```

3. **Download NLTK data:**

    ```sh
    python -m nltk.downloader vader_lexicon
    ```

4. **Download Spacy model:**

    ```sh
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Place your video file in the project directory**

2. **Run the script:**

    ```sh
    python script.py <filename>
    ```

    Or, you can use the python notebook: `create-gif.ipynb`.

3. **Output:**

    The script will create several GIFs with captions based on the selected sentences from the video. The GIFs will be saved in the project directory with filenames `output_gif_0.gif`, `output_gif_1.gif`, etc.

## How It Works

1. **Transcription:**

    The `transcribe_video` function uses OpenAI Whisper to transcribe the video and obtain word timestamps.

2. **Sentence Selection:**

    The `select_suitable_dialogues` function applies heuristics to determine if a sentence is suitable for creating a GIF. The heuristics include checking for named entities, action verbs, emotional content, and sentence length.

3. **GIF Creation:**

    The `create_gif_with_caption` function extracts video segments corresponding to the selected sentences and creates GIFs with captions.

## Example

Suppose you have a video named `long-vid.mp4`. Place it in the project directory and run the script. The script will generate GIFs based on the transcribed sentences and save them in the project directory.
