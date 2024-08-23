import gradio as gr
from src.transcriber import transcriber

def main():
    with gr.Blocks(title='multilang-asr-transcriber') as demo:
        gr.Markdown('# multilang-asr-transcriber')
        gr.Markdown('### A multilingual automatic speech transcription tool using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Supports translation to english and user setting of max words per line.')
        video_file = gr.File(file_types=["video"],type="filepath", label="Upload a video")
        max_words_per_line = gr.Number(value=6, label="Max words per line")
        task = gr.Dropdown(choices=["transcribe", "translate"], value="transcribe", label="Select Task")
        text_output = gr.Textbox(label="Text transcription")
        srt_file = gr.File(file_count="single", type="filepath", file_types=[".srt"], label="SRT file")
        gr.Interface(transcriber,
                    inputs=[video_file, max_words_per_line, task],
                    outputs=[text_output,srt_file],
                    allow_flagging="never")
    demo.launch(allowed_paths=["https://marquesafonso-multilang-asr-transcriber.hf.space/file=/tmp/gradio/", "file=/tmp/gradio/", "/tmp/gradio/"])

if __name__ == '__main__':
    main()