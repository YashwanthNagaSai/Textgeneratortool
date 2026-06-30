import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate(prompt):
    return generator(prompt, max_length=40)[0]["generated_text"]

demo = gr.Interface(
    fn=generate,
    inputs="text",
    outputs="text"
)

if __name__ == "__main__":
    demo.launch()