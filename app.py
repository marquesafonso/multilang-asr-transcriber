import gradio as gr
from src.transcriber import transcriber

def main():    
    with gr.Blocks(title='multilang-asr-transcriber', delete_cache=(86400, 86400), theme=gr.themes.Base()) as demo:
        gr.Markdown('## Multilang ASR Transcriber')
        gr.Markdown('An automatic speech recognition tool using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Supports multilingual video transcription and translation to english. Users may set the max words per line.')
        with gr.Tabs(selected="video") as tabs:
            with gr.Tab("Video", id="video"):
                file = gr.File(file_types=["video"],type="filepath", label="Upload a video")
                file_type = gr.Radio(value="video", choices=["video"], value="video", label="File Type", visible=False)
                max_words_per_line = gr.Number(value=6, label="Max words per line")
                task = gr.Radio(choices=["transcribe", "translate"], value="transcribe", label="Select Task")
                model_version = gr.Radio(choices=["deepdml/faster-whisper-large-v3-turbo-ct2",
                                                "turbo",
                                                "large-v3"], value="deepdml/faster-whisper-large-v3-turbo-ct2", label="Select Model")
                text_output = gr.Textbox(label="SRT Text transcription")
                srt_file = gr.File(file_count="single", type="filepath", file_types=[".srt"], label="SRT file")
                text_clean_output = gr.Textbox(label="Text transcription")
                gr.Interface(transcriber,
                            inputs=[file, file_type, max_words_per_line, task, model_version],
                            outputs=[text_output, srt_file, text_clean_output],
                            allow_flagging="never")
            
            with gr.Tab("Audio", id = "audio"):
                file = gr.File(file_types=["audio"],type="filepath", label="Upload an audio file")
                file_type = gr.Radio(value="audio", choices=["audio"], value="audio", label="File Type", visible=False)
                max_words_per_line = gr.Number(value=6, label="Max words per line")
                task = gr.Radio(choices=["transcribe", "translate"], value="transcribe", label="Select Task")
                model_version = gr.Radio(choices=["deepdml/faster-whisper-large-v3-turbo-ct2",
                                                "turbo",
                                                "large-v3"], value="deepdml/faster-whisper-large-v3-turbo-ct2", label="Select Model")
                text_output = gr.Textbox(label="SRT Text transcription")
                srt_file = gr.File(file_count="single", type="filepath", file_types=[".srt"], label="SRT file")
                text_clean_output = gr.Textbox(label="Text transcription")
                gr.Interface(transcriber,
                            inputs=[file, file_type, max_words_per_line, task, model_version],
                            outputs=[text_output, srt_file, text_clean_output],
                            allow_flagging="never")
    demo.launch()

if __name__ == '__main__':
    main()