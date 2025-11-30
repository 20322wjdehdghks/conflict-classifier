import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ----------------------
# 1. 모델 로드
# ----------------------
@st.cache_resource
def load_conflict_model():
    return load_model('conflict_classifier.h5')

model = load_conflict_model()
labels = ['civil_war', 'international_war', 'protest', 'peace_meeting']

# ----------------------
# 2. 앱 UI
# ----------------------
st.title("국제 분쟁 이미지 분류 AI")
st.write("국제 분쟁 이미지를 업로드하면 AI가 유형을 예측합니다.")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg','png','jpeg'])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------
    # 3. 예측
    # ----------------------
    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.image(img, caption=f"예측된 분쟁 유형: {predicted_label} ({confidence:.2f}%)", use_column_width=True)
