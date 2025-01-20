# Real-Time Emotion Recognition

A real-time emotion recognition system that combines deep learning for emotion classification with MediaPipe for facial landmark detection. The system provides instant feedback on detected emotions with probability distributions and facial mesh visualization.

![Example Output](example_output.png)

## Features

- **Real-time Processing**: Instant emotion detection from webcam feed
- **Multi-feature Detection**:
  - Face detection and tracking
  - Facial landmark visualization (468 points)
  - Emotion classification with confidence scores
- **Advanced Visualization**:
  - Facial mesh overlay
  - Primary emotion with confidence level
  - Distribution of all detected emotions
- **Smooth Predictions**: Temporal averaging for stable output

## Technology Stack

- **Deep Learning**: PyTorch for emotion classification
- **Computer Vision**: OpenCV and MediaPipe for face detection and mesh generation
- **Model Architecture**: Custom implementation combining:
  - ResNet-50 backbone
  - Local feature extraction
  - Multi-level attention mechanisms
  - Feature fusion for final classification

## Detected Emotions

The system detects seven basic emotions:
- Neutral
- Happiness
- Sadness
- Surprise
- Fear
- Disgust
- Anger

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.5.1
mediapipe>=0.8.9
numpy>=1.19.2
```

## Usage
Controls:
- Press 'q' to exit the application
- The system will automatically begin detecting faces and emotions when launched

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── efficient_face.py     # Main model architecture
│   │   ├── attention.py          # Attention mechanism
│   │   └── local_feature.py      # Local feature extractor
│   └── utils/
│       ├── visualization.py      # Visualization functions
│       └── camera.py            # Camera handling
├── models/
│   └── model_weights.pth        # Pretrained weights
└── main.py                      # Application entry point
```

## Performance

The system achieves:
- Real-time processing on standard CPU
- Enhanced performance with GPU acceleration
- Stable emotion detection with temporal smoothing

## Future Improvements

- [ ] Multi-face tracking and emotion detection
- [ ] Emotion statistics and logging
- [ ] Custom GUI interface
- [ ] Mobile device optimization
- [ ] Additional facial analysis features

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.
