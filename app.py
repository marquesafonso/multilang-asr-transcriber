import gradio as gr
from src.transcriber import transcriber

def main():
    with gr.Blocks(title='multilang-asr-transcriber', delete_cache=(86400, 86400), theme=gr.themes.Ocean()) as demo:
        gr.Markdown('## Multilang ASR Transcriber')
        gr.Markdown('An automatic speech recognition tool using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Supports multilingual video transcription and translation to english. Users may set the max words per line.')
        video_file = gr.File(file_types=["video"],type="filepath", label="Upload a video")
        max_words_per_line = gr.Number(value=6, label="Max words per line")
        task = gr.Dropdown(choices=["transcribe", "translate"], value="transcribe", label="Select Task")
        text_output = gr.Textbox(label="SRT Text transcription", show_copy_button=True)
        srt_file = gr.File(file_count="single", type="filepath", file_types=[".srt"], label="SRT file")
        text_clean_output = gr.Textbox(label="Text transcription", show_copy_button=True)
        gr.Interface(transcriber,
                    inputs=[video_file, max_words_per_line, task],
                    outputs=[text_output, srt_file, text_clean_output],
                    allow_flagging="never")
    demo.launch()

if __name__ == '__main__':
    main()