import streamlit as st
from PIL import Image, ImageDraw
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


def draw_boxes(image, detections, target_class=None):
    draw = ImageDraw.Draw(image)

    for detection in detections:
        label = detection['label']
        score = detection['score']
        box = detection['box']

        if target_class and label != target_class:
            continue

        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        text = f"{label}: {score:.2f}"
        text_bbox = draw.textbbox((xmin, ymin), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle([xmin, ymin, xmin + text_width + 6, ymin + text_height + 6], fill="red")

        draw.text((xmin + 3, ymin + 3), text, fill="white")

    return image


yolo_classes, detector = load_model_and_classes()

uploaded_file = st.file_uploader("Выберите изображение", type=['jpg', 'png', 'jpeg'])
selected_class = st.selectbox("Выберите класс для обнаружения:", [""] + yolo_classes)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ваше изображение", use_container_width=True)
    if st.button("Обнаружить объекты"):
        if detector is None:
            st.error("Модель не загружена!")
        else:
            with st.spinner("Обрабатываем изображение..."):
                try:
                    detections = detector(image)
                    if selected_class:
                        filtered_detections = [det for det in detections if det['label'] == selected_class]
                        if not filtered_detections:
                            st.warning(f"Объекты класса '{selected_class}' не обнаружены на изображении")
                    else:
                        filtered_detections = detections
                    image_with_boxes = image.copy()
                    image_with_boxes = draw_boxes(image_with_boxes, filtered_detections, selected_class)
                    st.image(image_with_boxes, caption="Результат обнаружения", use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка при обработке изображения: {e}")
