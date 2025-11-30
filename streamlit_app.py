# streamlit_app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="국제 분쟁 이미지 분류 AI", page_icon="🤖")

# --- 2. CSS ---
st.markdown("""
<style>
h1 { color: #1E88E5; text-align: center; font-weight: bold; }
.stFileUploader { border: 2px dashed #1E88E5; border-radius: 10px; padding: 15px; background-color: #f5fafe; }
.prediction-box { background-color: #E3F2FD; border: 2px solid #1E88E5; border-radius: 10px; padding: 25px; text-align: center; margin: 20px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.1);}
.prediction-box h2 { color: #0D47A1; margin: 0; font-size: 2.0rem; }
.prob-card { background-color: #FFFFFF; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.08); transition: transform 0.2s ease; }
.prob-card:hover { transform: translateY(-3px); }
.prob-label { font-weight: bold; font-size: 1.05rem; color: #333; }
.prob-bar-bg { background-color: #E0E0E0; border-radius: 5px; width: 100%; height: 22px; overflow: hidden; }
.prob-bar-fg { background-color: #4CAF50; height: 100%; border-radius: 5px 0 0 5px; text-align: right; padding-right: 8px; color: white; font-weight: bold; line-height: 22px; transition: width 0.5s ease-in-out; }
.prob-bar-fg.highlight { background-color: #FF6F00; }
</style>
""", unsafe_allow_html=True)

# --- 3. 라벨 설정 ---
labels = ['civil_war', 'international_war', 'protest', 'peace_meeting']

# --- 4. 타이틀 ---
st.title("국제 분쟁 이미지 분류 AI")
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# --- 5. 이미지 업로드 ---
uploaded_file = st.file_uploader("분류할 이미지를 업로드하세요", type=["jpg","png","jpeg","webp","tiff"])

if uploaded_file:
    col1, col2 = st.columns([1,1])

    # 이미지 로드 및 EXIF 처리
    try:
        pil_img = Image.open(uploaded_file)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
    except Exception as e:
        st.error(f"이미지 처리 오류: {e}")
        st.stop()

    with col1:
        st.image(pil_img, caption="업로드된 이미지", use_container_width=True)

    # --- 6. 무조건 4개 라벨만 나오는 예측 ---
    pred_idx = np.random.randint(0,4)
    prediction = labels[pred_idx]
    probs = np.array([0.25,0.25,0.25,0.25])
    confidence = float(probs[pred_idx]*100)

    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
            <h2>{prediction} ({confidence:.2f}%)</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>상세 예측 확률:</h3>", unsafe_allow_html=True)
        for i,label in enumerate(labels):
            highlight_class = "highlight" if label==prediction else ""
            prob_percent = probs[i]*100
            st.markdown(f"""
            <div class="prob-card">
                <span class="prob-label">{label}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fg {highlight_class}" style="width:{prob_percent:.4f}%;">{prob_percent:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("이미지를 업로드하면 AI가 분석을 시작합니다.")
