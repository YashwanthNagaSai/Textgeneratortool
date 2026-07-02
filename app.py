from transformers import pipeline
import gradio as gr

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_text(prompt):
    result = generator(
        prompt,
        max_length=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter your prompt..."
    ),
    outputs=gr.Textbox(lines=8),
    title="🤖 AI Text Generator",
    description="Generate text using GPT-2"
)

demo.launch()