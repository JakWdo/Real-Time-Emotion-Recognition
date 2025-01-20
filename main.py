# main.py

import torch
import cv2
import mediapipe as mp
from src.models.efficient_face import EfficientFaceResNet
from src.utils.camera import CameraHandler
from src.utils.visualization import Visualizer

# Definicja etykiet emocji
EMOTION_LABELS = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']

def main():
    # Inicjalizacja urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Inicjalizacja modelu
    model = EfficientFaceResNet(num_classes=len(EMOTION_LABELS))
    model.load_state_dict(torch.load('models/model_weights.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Inicjalizacja obsługi kamery i wizualizacji
    camera_handler = CameraHandler()
    visualizer = Visualizer(EMOTION_LABELS)

    print("Starting emotion recognition... Press 'q' to quit")
    
    try:
        while True:
            # Przechwycenie i przetworzenie klatki
            frame = camera_handler.get_frame()
            if frame is None:
                break

            # Detekcja twarzy i predykcja emocji
            faces = camera_handler.detect_faces(frame)
            for face_data in faces:
                # Przygotowanie obrazu twarzy
                face_tensor = camera_handler.preprocess_face(face_data['face'])
                face_tensor = face_tensor.to(device)

                # Predykcja emocji
                with torch.no_grad():
                    output = model(face_tensor)
                    output = output.cpu().numpy()[0]

                # Aktualizacja bufora predykcji i wizualizacja
                camera_handler.update_prediction_buffer(output)
                smoothed_output = camera_handler.get_smoothed_prediction()
                
                if smoothed_output is not None:
                    visualizer.draw_results(frame, face_data, smoothed_output)

            # Wyświetlenie wyniku
            cv2.imshow('Real-Time Emotion Recognition', frame)

            # Wyjście po naciśnięciu 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera_handler.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
