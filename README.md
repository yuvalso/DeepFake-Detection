# DeepFake Detection

This project is a web-based DeepFake detection system that uses machine learning models to classify videos as real or fake, with visual explanations using Grad-CAM overlays and attention maps.

# Features

- DeepFake classification using ResNeXt and LSTM models
- Face detection with OpenCV
- Grad-CAM & attention map overlays to visualize model focus
- Web interface built with Flask
- Option to choose between a pre-trained model and a custom-trained model

# How It Works

1. Upload a video through the web interface.
2. The system extracts faces from video frames.
3. Each face is analyzed using a deep learning model.
4. Grad-CAM overlays show which parts of the face influenced the decision.
5. A final verdict (real/fake) is shown to the user.
