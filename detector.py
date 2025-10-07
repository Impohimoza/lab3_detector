import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("Detector v1")

@st.cache_resource
def load_model_and_classes():

    try:
        detector = pipeline("object-detection", model="hustvl/yolos-tiny")
    
        if hasattr(detector.model.config, 'id2label') and detector.model.config.id2label:
            classes = [
                    class_name for class_name in detector.model.config.id2label.values() 
                    if class_name != "N/A" and class_name is not None
                ]
            return classes, detector
        else:
            st.error("Не удалось получить классы из модели")
            return [], None

    except Exception as e:       
        st.error(f"Ошибка загрузки модели: {e}")
        return [], None

yolo_classes, detector  = load_model_and_classes()

uploaded_file = st.file_uploader("Выберите изображение", type=['jpg', 'png', 'jpeg'])
selected_class = st.selectbox("Выберите класс для обнаружения:", [""] + yolo_classes)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ваше изображение", use_container_width = True)

    # if selected_class and selected_class != "":
       