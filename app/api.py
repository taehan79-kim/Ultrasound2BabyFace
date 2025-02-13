from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from generate import generate_image, load_models  # 이 부분은 기존 코드에서 사용한 generate_image 함수
import io
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the depth_estimator, feature_extractor, pipe
    depth_estimator, feature_extractor, pipe = load_models()

    # 모델 객체를 app.state에 저장하여 앱 종료 시 접근 가능하도록 함
    app.state.depth_estimator = depth_estimator
    app.state.feature_extractor = feature_extractor
    app.state.pipe = pipe

    yield

    # 앱 종료 시 모델 리소스 해제
    app.state.depth_estimator = None
    app.state.feature_extractor = None
    app.state.pipe = None

app = FastAPI(lifespan=lifespan)

@app.post("/generate-image/")
async def generate_image_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(None),   # 텍스트 값
    neg_prompt: str = Form(None),   # 텍스트 값
    seed: int = Form(None),     # 정수 값
    strength: float = Form(None)
    ):
    # 이미지 파일을 받아서 처리
    image = Image.open(io.BytesIO(await file.read()))
    
    # app.state에서 모델 객체 가져오기
    depth_estimator = app.state.depth_estimator
    feature_extractor = app.state.feature_extractor
    pipe = app.state.pipe

    # generate_image 함수로 이미지 처리
    result_image = generate_image(depth_estimator, feature_extractor, pipe, image, prompt, neg_prompt, seed, strength)
    
    # 결과 이미지 바이트로 변환
    img_byte_arr = BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

# CORS 미들웨어 설정
# 프론트엔드(localhost:5174)에서의 API 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('FRONTEND_URL', 'http://localhost:5173')], # 프론트엔드 서버 주소
    allow_credentials=True,                  # 자격 증명(쿠키 등) 허용
    allow_methods=["*"],                     # 모든 HTTP 메서드 허용
    allow_headers=["*"],                     # 모든 HTTP 헤더 허용
)

# uvicorn 실행 코드 추가
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)