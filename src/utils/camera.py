# src/utils/camera.py

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from collections import deque

class CameraHandler:
    def __init__(self, buffer_size=5):
        # Inicjalizacja kamery
        self.cap = cv2.VideoCapture(0)
        
        # Inicjalizacja MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Bufor predykcji
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # Transformacje obrazu
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_frame(self):
        """Pobiera klatkę z kamery"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def detect_faces(self, frame):
        """
        Wykrywa twarze na klatce i zwraca ich dane.
        
        Args:
            frame: Klatka wideo do analizy
            
        Returns:
            Lista słowników zawierających dane wykrytych twarzy
            (obszar twarzy, współrzędne bounding box, punkty charakterystyczne)
        """
        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detekcja twarzy
        detections = self.face_detection.process(rgb_frame)
        if not detections.detections:
            return results

        # Detekcja punktów charakterystycznych
        mesh_results = self.face_mesh.process(rgb_frame)
        
        for detection in detections.detections:
            # Pobranie współrzędnych twarzy
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            
            # Zabezpieczenie przed wyjściem poza granice obrazu
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            # Wycięcie obszaru twarzy
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
                
            results.append({
                'face': face,
                'bbox': (x, y, w, h),
                'mesh': mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
            })
            
        return results

    def preprocess_face(self, face):
        """
        Przygotowuje obraz twarzy do predykcji.
        
        Args:
            face: Wycięty obszar twarzy z klatki
            
        Returns:
            Tensor przygotowany do podania na wejście modelu
        """
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(face_rgb).unsqueeze(0)
        return face_tensor

    def update_prediction_buffer(self, prediction):
        """
        Aktualizuje bufor predykcji o nową wartość.
        
        Args:
            prediction: Nowa predykcja do dodania do bufora
        """
        self.prediction_buffer.append(prediction)

    def get_smoothed_prediction(self):
        """
        Zwraca uśrednioną predykcję z bufora.
        
        Returns:
            Uśredniona predykcja lub None jeśli bufor jest pusty
        """
        if len(self.prediction_buffer) == 0:
            return None
        return np.mean(self.prediction_buffer, axis=0)

    def release(self):
        """Zwalnia zasoby kamery"""
        self.cap.release()
