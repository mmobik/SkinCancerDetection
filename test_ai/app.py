# Точка входа в программу

from predict import load_model
from interface import create_gradio_interface

if __name__ == '__main__':
    # Путь к модели
    MODEL_PATH = 'saved_model/improved_hmnist.pth'
    model = load_model(MODEL_PATH)  # Загружаем модель
    gradio_interface = create_gradio_interface(model)  # Создаем интерфейс
    gradio_interface.launch(share=False)  # Запускаем приложение
