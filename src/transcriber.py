import os, json
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

def write_srt(segments, max_words_per_line, srt_path, device_type):

    # Pause and char heuristics
    max_chars = 23 if device_type == "mobile" else 42
    pause_threshold = 2.0

    with open(srt_path, "w", encoding="utf-8") as file:
        result = ""
        result_clean = []
        json_output = {"lines": []}
        line_counter = 1

        words_in_line = []

        for segment in segments:
            for word in segment.words:
                # Check if adding this word breaks char limit
                tentative_line = " ".join([w.word.strip() for w in words_in_line + [word]])

                # Detect pause (gap from previous word)
                long_pause = False
                if words_in_line:
                    prev_word = words_in_line[-1]
                    if word.start - prev_word.end >= pause_threshold:
                        long_pause = True

                word_overflow = len(words_in_line) >= max_words_per_line
                char_overflow = len(tentative_line) > max_chars
                # Break conditions
                if (word_overflow or char_overflow or long_pause):
                    # Finalize current line
                    if words_in_line:
                        start_time = convert_seconds_to_time(words_in_line[0].start)
                        end_time = convert_seconds_to_time(words_in_line[-1].end)
                        line_text = " ".join([w.word.strip() for w in words_in_line])

                        # SRT
                        result += f"{line_counter}\n{start_time} --> {end_time}\n{line_text}\n\n"
                        result_clean.append(line_text)

                        # JSON
                        json_output["lines"].append({
                            "line_index": line_counter,
                            "start": words_in_line[0].start,
                            "end": words_in_line[-1].end,
                            "text": line_text,
                            "words": [
                                {"word": w.word.strip(), "start": w.start, "end": w.end}
                                for w in words_in_line
                            ]
                        })
                        line_counter += 1

                    # Start a fresh line with the current word
                    words_in_line = [word]
                else:
                    # keep adding words
                    words_in_line.append(word)

        # Flush last line
        if words_in_line:
            start_time = convert_seconds_to_time(words_in_line[0].start)
            end_time = convert_seconds_to_time(words_in_line[-1].end)
            line_text = " ".join([w.word.strip() for w in words_in_line])

            result += f"{line_counter}\n{start_time} --> {end_time}\n{line_text}\n\n"
            result_clean.append(line_text)

            json_output["lines"].append({
                "line_index": line_counter,
                "start": words_in_line[0].start,
                "end": words_in_line[-1].end,
                "text": line_text,
                "words": [
                    {"word": w.word.strip(), "start": w.start, "end": w.end}
                    for w in words_in_line
                ]
            })

        file.write(result)
        return result, srt_path, " ".join(result_clean), json.dumps(json_output)


def transcriber(file_input:gr.File,
                file_type: str,
                max_words_per_line:int,
                task:str,
                model_version:str,
                device_type: str):
    srt_filepath = os.path.normpath(f"{file_input.split('.')[0]}.srt")
    if file_type == "video" :
        audio_input = convert_video_to_audio(file_input)
    else:
        audio_input = file_input
    model = WhisperModel(model_version, device="auto", compute_type="int8")
    segments, _ = model.transcribe(
        audio_input,
        beam_size=5,
        task=task,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )
    return write_srt(segments=segments, max_words_per_line=max_words_per_line, srt_path=srt_filepath, device_type=device_type)