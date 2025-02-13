import gradio as gr
from scripts.generate import generate_image
from PIL import Image

def process_image(image):
    result = generate_image(image)
    return result

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Ultrasound to Baby Face Generator",
    description="Upload an ultrasound image to generate a baby face.",
)

demo.launch(share=True)