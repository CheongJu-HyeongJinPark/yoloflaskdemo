import pathlib
import uuid
from typing import Any

import pandas as pd
import torch
from app.codes import codes_to_text, get_disease_name
from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from PIL import Image
from werkzeug import Response as BaseResponse

app = Flask(__name__, template_folder="../templates")

# 모델 파일 경로
model_path = pathlib.Path(__file__).parent.parent / "model.pt"

# 바운딩 박스 이미지 저장 위치
image_dir = pathlib.Path(__file__).parent.parent / "images"

def render_dataframes_to_text(crop_name: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    pandas.DataFrame 객체 중 인식된 [name] 값을 추출합니다.
    추출한 [name] 값을 "-" 기준으로 나누어 area, disease, risk, grow 코드를 추출합니다.
    추출한 코드를 codes_to_text 함수에서 사람이 읽을 수 있는 문장으로 변경처리를 합니다.
    """

    # DataFrame 값이 비어있는 경우
    if df.empty:
        return {"title": "인식 실패", "pesticide": []}

    # 첫번째 값을 가져옵니다
    name = df.get("name")[0]

    # xx-xx-xx-xx 형태의 값을 "-"을 기준으로 나눕니다
    area_code, disease_code, risk_code, grow_code = name.split("-")

    # 병해충명을 가져옵니다
    disease = get_disease_name(disease_code)

    return {
        # 사용자에게 노출되는 내용
        "title": codes_to_text(
            area_code=area_code, disease_code=disease_code, risk_code=risk_code, grow_code=grow_code
        )
    }


@app.route("/", methods=["GET"])
def index() -> str:
    """사용자가 이미지를 업로드하는 페이지를 렌더링하는 엔드포인트입니다."""
    return render_template("index.html", uuid=uuid.uuid4())


@app.route("/predict", methods=["POST"])
def predict() -> str | BaseResponse:
    """
    YoloV5 모델을 통해 Object Detection 을 진행한 후 predict.html 을 렌더링합니다.
    """
    name = request.form.get("name")
    crop_name = request.form.get("cropName")
    # 사용자가 업로드한 파일 객체를 불러옵니다
    file = request.files.get("image")
    if not name or not crop_name or not file:  # 해당 파일이 존재하지 않는 경우 다시 index 로 돌려보냅니다
        return redirect(url_for("index"))

    # 이미지를 Pillow 객체로 변환합니다
    image = Image.open(file.stream)
    # 이미지를 640x640 사이즈로 변환합니다
    resized_image = image.resize((640, 640))
    # 리사이징된 이미지를 모델에 입력합니다
    results = model([resized_image])

    # 결과 값에 박스와 레이블을 추가합니다
    results.render()

    # Yolo Detection 결과물을 pandas.DataFrame 으로 변환합니다
    df: pd.DataFrame = results.pandas().xyxy[0]
    context = render_dataframes_to_text(crop_name, df)

    # 이미지를 images 폴더에 저장합니다
    Image.fromarray(results.ims[0]).save(image_dir / f"{name}.png", "PNG")

    return render_template("predict.html", **context, name=f"{name}.png")


@app.route("/images/<path:path>")
def images(path: str) -> BaseResponse:
    """저장된 바운딩 박스 이미지를 불러오기 위한 경로입니다"""
    return send_from_directory(image_dir, path)


if __name__ == "__main__":
    # YOLOv5 와 가중치 모델을 불러들입니다
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    model.eval()

    # Flask를 실행합니다
    app.run(port=9000, debug=True)

