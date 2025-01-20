# src/utils/visualization.py

import cv2
import numpy as np
import mediapipe as mp

class Visualizer:
    def __init__(self, emotion_labels):
        """
        Inicjalizacja wizualizatora.
        
        Args:
            emotion_labels: Lista etykiet emocji
        """
        self.emotion_labels = emotion_labels
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

    def draw_results(self, frame, face_data, predictions):
        """
        Rysuje wyniki detekcji i predykcji na klatce.
        
        Args:
            frame: Klatka wideo
            face_data: Słownik z danymi twarzy (bbox, mesh)
            predictions: Tablica prawdopodobieństw emocji
        """
        x, y, w, h = face_data['bbox']
        mesh = face_data['mesh']

        # Rysowanie siatki punktów charakterystycznych
        if mesh:
            self._draw_face_mesh(frame, mesh)

        # Konwersja log_softmax do prawdopodobieństw
        probabilities = np.exp(predictions)
        probabilities = probabilities / np.sum(probabilities) * 100

        # Rysowanie wyników
        self._draw_main_emotion(frame, probabilities, x, y)
        self._draw_all_probabilities(frame, probabilities, x, y, h)

    def _draw_face_mesh(self, frame, mesh):
        """
        Rysuje siatkę punktów charakterystycznych twarzy.
        
        Args:
            frame: Klatka wideo
            mesh: Punkty charakterystyczne twarzy z MediaPipe
        """
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=mesh,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=1,
                circle_radius=1
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 0, 255),
                thickness=1,
                circle_radius=1
            )
        )

    def _draw_main_emotion(self, frame, probabilities, x, y):
        """
        Rysuje główną wykrytą emocję.
        
        Args:
            frame: Klatka wideo
            probabilities: Tablica prawdopodobieństw
            x, y: Współrzędne do umieszczenia tekstu
        """
        main_emotion_idx = np.argmax(probabilities)
        main_emotion = self.emotion_labels[main_emotion_idx]
        main_prob = probabilities[main_emotion_idx]

        text = f"{main_emotion}: {main_prob:.1f}%"
        cv2.putText(
            frame, text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2,
            cv2.LINE_AA
        )

    def _draw_all_probabilities(self, frame, probabilities, x, y, h):
        """
        Rysuje prawdopodobieństwa wszystkich emocji.
        
        Args:
            frame: Klatka wideo
            probabilities: Tablica prawdopodobieństw
            x, y, h: Współrzędne i wysokość do umieszczenia tekstu
        """
        # Tworzenie listy emocji z prawdopodobieństwami (bez głównej emocji)
        main_idx = np.argmax(probabilities)
        other_emotions = [
            f"{self.emotion_labels[i]}: {prob:.1f}%"
            for i, prob in enumerate(probabilities)
            if i != main_idx
        ]
        
        # Łączenie tekstu i wyświetlanie
        other_text = " / ".join(other_emotions)
        cv2.putText(
            frame, other_text,
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 0, 0), 1,
            cv2.LINE_AA
        )
