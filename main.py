import sys
import whisper
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download necessary NLTK data
nltk.download('vader_lexicon')

def compute_lps_array(pattern):
    """Compute the Longest Prefix Suffix (LPS) array for KMP algorithm."""
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(sentence, words_list):
    """Search for a sentence in the list of words using KMP algorithm."""
    
    # Preprocess sentence into list of words
    pattern = sentence.lower().split()
    text = [word["word"].lower() for word in words_list]

    # Compute LPS array
    lps = compute_lps_array(pattern)

    i = 0  # index for text
    j = 0  # index for pattern

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == len(pattern):
            j = lps[j - 1]
            return i - len(pattern)  # Return the starting index of the match
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return -1  # If no match is found

def match_sentences_with_words(sentences, words):
    """Find sentences in the list of words using KMP algorithm."""
    results = {}
    for sentence in sentences:
        matches = kmp_search(sentence.strip(), words)
        results[sentence.strip()] = matches
    return results

def transcribe_video(video_path):
    """Transcribe the video using Whisper."""
    model = whisper.load_model("small.en")
    result = model.transcribe(video_path, word_timestamps=True)
    return result

def is_gif_worthy(text, nlp, sia):
    """Determine if a text segment is suitable for a GIF using few heuristics."""
    doc = nlp(text)
    
    # Check for named entities (people, places, organizations)
    has_named_entity = any(ent.label_ in ['PERSON', 'ORG', 'GPE'] for ent in doc.ents)
    
    # Check for action verbs
    has_action_verb = any(token.pos_ == 'VERB' and token.dep_ == 'ROOT' for token in doc)
    
    # Analyze sentiment
    sentiment_scores = sia.polarity_scores(text)
    is_emotional = abs(sentiment_scores['compound']) > 0.2
    
    # Check for questions or exclamations
    is_question_or_exclamation = any(sent.text.strip().endswith(('?', '!')) for sent in doc.sents)
    
    # Combine criteria
    return (has_named_entity or has_action_verb or is_emotional or is_question_or_exclamation) and len(text.split()) <= 6

def select_suitable_dialogues(segments, nlp, sia):
    suitable_sentences = []
    for segment in segments:
        if is_gif_worthy(segment, nlp, sia):
            suitable_sentences.append(segment)
    return suitable_sentences

def create_gif_with_caption(video_path, start_time, end_time, text, output_path):
    video = mp.VideoFileClip(video_path).subclip(start_time, end_time)
    
    txt_clip = mp.TextClip(text, fontsize=128, color='white', font='Arial',
                            stroke_color='black', stroke_width=2)
    txt_clip = txt_clip.set_pos('bottom').set_duration(video.duration)
    final_clip = mp.CompositeVideoClip([video, txt_clip])
    final_clip.write_gif(output_path, fps=10)

def main(video_path):
    # Load NLP models
    nlp = spacy.load("en_core_web_sm")
    sia = SentimentIntensityAnalyzer()
    
    # Transcribe the video
    result = transcribe_video(video_path)

    # Select sentences suitable for GIFs
    sentences = result["text"].split(".")
    suitable_sentences = select_suitable_dialogues(sentences, nlp, sia)

    words = []
    for segment in result["segments"]:
        for word in segment['words']:
            if(word['word'][-1] == '.'):
                word['word'] = word['word'][:-1]
            words.append({'word': word['word'].strip(), 'start': word['start'], 'end': word['end']})

    # match sentences with words using KMP algorithm
    sentence_matches = match_sentences_with_words(suitable_sentences, words)

    suitable_segments = []
    for sentence, match in sentence_matches.items():
        if match != -1:
            start_time = words[match]['start']
            end_time = words[match + len(sentence.split()) - 1]['end']
            suitable_segments.append({"text": sentence, "start": start_time, "end": end_time})

    
    # Create GIFs with captions
    for i, segment in enumerate(suitable_segments):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # Clean up text (remove newlines, extra spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        
        output_path = f"output_gif_{i}.gif"
        create_gif_with_caption(video_path, start_time, end_time, text, output_path)
        print(f"Created GIF: {output_path}")

if __name__ == "__main__":
    video_path = "long-vid.mp4"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    main(video_path)