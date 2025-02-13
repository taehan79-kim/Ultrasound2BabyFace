import gradio as gr
import requests
from PIL import Image
from io import BytesIO

def process_inputs(image, text1, text2, seed1, seed2):
    # 이미지 데이터를 바이트로 변환
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # 서버로 보낼 데이터 설정
    url = os.getenv('GENERATE_URL')  # POST 요청을 보낼 URL
    files = {"file": ("image.png", image_bytes, "image/png")}
    data = {"prompt": text1, "neg_prompt": text2, "seed": seed1, "strength": seed2}
    
    # POST 요청 보내기
    response = requests.post(url, files=files, data=data)
    
    # 서버에서 받은 응답 처리 (이미지 반환을 가정)
    if response.status_code == 200:
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        return img
    else:
        return "Error: Failed to generate image"

demo = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),  # 이미지 입력
        gr.Textbox(value="A professional portrait of an adorable Asian baby sleeping, peaceful, serene, soothing, closed eyes, eyelashes, 8k photo", label="prompt"),   # 텍스트 입력
        gr.Textbox(value="poorly rendered face, poorly drawn face, poor facial details, blurry image, bad anatomy", label="negativeprompt"),
        gr.Number(value=123456772, label="Seed", precision=0),  # 첫 번째 정수 입력
        gr.Slider(value=0.7, minimum=0, maximum=1, step=0.1, label="strength")   # 두 번째 정수 입력
    ],
    outputs=gr.Image(type="pil"),
    title="Ultrasound to Baby Face Generator",
    description="Upload an ultrasound image to generate a baby face.",
)

demo.launch(share=True)