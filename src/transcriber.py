import os
import gradio as gr
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip

def convert_video_to_audio(video_input):
    video_clip = VideoFileClip(video_input)
    audio_clip = video_clip.audio
    audio_clip_filepath = os.path.normpath(f"{video_input.split('.')[0]}.m4a")
    audio_clip.write_audiofile(audio_clip_filepath, codec='aac')
    audio_clip.close()
    video_clip.close()
    return audio_clip_filepath

def convert_seconds_to_time(seconds):
    seconds = float(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    whole_seconds = int(remainder)
    milliseconds = int((remainder - whole_seconds) * 1000)    
    return f"{int(hours):02}:{int(minutes):02}:{whole_seconds:02},{milliseconds:03}"

def write_srt(segments, max_words_per_line, srt_path):
    with open(srt_path, "w", encoding='utf-8') as file:
        result = ''
        line_counter = 1
        for _, segment in enumerate(segments):
            words_in_line = []
            for w, word in enumerate(segment.words):
                words_in_line.append(word)
                # Write the line if max words limit reached or it's the last word in the segment
                if len(words_in_line) == max_words_per_line or w == len(segment.words) - 1:
                    if words_in_line:  # Check to avoid writing a line if there are no words
                        start_time = convert_seconds_to_time(words_in_line[0].start)
                        end_time = convert_seconds_to_time(words_in_line[-1].end)
                        line_text = ' '.join([w.word.strip() for w in words_in_line])
                        result += f"{line_counter}\n{start_time} --> {end_time}\n{line_text}\n\n"
                        # Reset for the next line and increment line counter
                        line_counter += 1
                    words_in_line = []  # Reset words list for the next line
        file.write(result)
        return result, srt_path

def transcriber(video_input:gr.File,
                max_words_per_line:int,
                task:str):
    srt_filepath = os.path.normpath(f"{video_input.split('.')[0]}.srt")
    gr.set_static_paths(paths=[srt_filepath])
    audio_input = convert_video_to_audio(video_input)
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio_input,
        beam_size=5,
        task=task,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )
    return write_srt(segments=segments, max_words_per_line=max_words_per_line, srt_path=srt_filepath)