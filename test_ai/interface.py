# Gradio-интерфейс
import gradio as gr
import os
from predict import predict_image

def create_gradio_interface(model):
    def process_image(img):  # Основная функция обработки изображения
        if img is None:
            return "No image uploaded"  # Проверка на отсутствие изображения
        temp_image_path = "temp_image.jpg"
        img.save(temp_image_path)  # Сохраняем изображение во временный файл
        predictions = predict_image(temp_image_path, model)  # Делаем предсказания
        os.remove(temp_image_path)  # Удаляем временный файл после обработки
        return predictions

    image_input = gr.Image(type="pil", label="Upload Mole Image")  # Ввод изображения через Gradio
    label = gr.Label(num_top_classes=3, label="Top 3 Predictions")  # Вывод предсказанных классов
    interface = gr.Interface(
        fn=process_image,
        inputs=image_input,
        outputs=label,
        title="Skin Cancer AI Detector",
        description="Upload an image of a mole to get a risk assessment.",
        live=False  # Без live-обработки
    )
    return interface
