from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []
features = []
gradients = []
app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# # For prediction of output  
# def predict(model, img, path='./'):
#   # use this command for gpu    
#   # fmap, logits = model(img.to('cuda'))
#   fmap, logits = model(img.to())
#   params = list(model.parameters())
#   weight_softmax = model.linear1.weight.detach().cpu().numpy()
#   logits = sm(logits)
#   _, prediction = torch.max(logits, 1)
#   confidence = logits[:, int(prediction.item())].item()*100
#   print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
#   return [int(prediction.item()), confidence]

##################### green

def predict(model, img, original_frame, path='./'):
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100

    # Backward pass to get gradients
    model.zero_grad()
    score = logits[:, prediction.item()]
    score.backward(retain_graph=True)

    # Generate CAM
    cam = generate_cam(features[0].detach().cpu(), gradients[0].detach().cpu())

    # Draw green rectangle(s) on the original frame
    face_locations = face_recognition.face_locations(original_frame)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(original_frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle

    #  Resize the CAM heatmap to match the original frame size
    cam = cv2.resize(cam, (original_frame.shape[1], original_frame.shape[0]))  # Width x Height
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Overlay the heatmap onto the original frame
    overlay = heatmap + np.float32(original_frame) / 255
    overlay = overlay / np.max(overlay)

    #  Save the output image
    output_path = os.path.join('Uploaded_Files', 'output_cam.png')
    cv2.imwrite(output_path, np.uint8(255 * overlay))

    print('confidence of prediction: ', confidence)
    return [int(prediction.item()), confidence, output_path]

###################### green !


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      faces = face_recognition.face_locations(frame)
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image


def detectFakeVideo(videoPath, model_choice="default"):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    path_to_videos = [videoPath]
    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)

    # Select model
    if model_choice == "custom":
        path_to_model = 'model/my_model.pt'  # 🔜 Add later
    else:
        path_to_model = 'model/df_model.pt'  # Using now

    model = Model(2)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
#################### green
    # Choose your target CNN layer (replace 'model.7' if needed)
    target_layer = model.model[-1]  # Example: last ResNeXt layer

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register the hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

################################## green !

    for i in range(len(path_to_videos)):
        ##prediction = predict(model, video_dataset[i], './')
        ###### green
        # Extract one original frame for visualization
        vidcap = cv2.VideoCapture(videoPath)
        success, frame = vidcap.read()
        vidcap.release()

        prediction = predict(model, video_dataset[i], frame, './')
        ###### green !
    return prediction


################# green


# def generate_cam(feature_map, gradients):
#     weights = np.mean(gradients, axis=(2, 3))[0, :]  # Global average pooling
#     cam = np.zeros(feature_map.shape[2:], dtype=np.float32)

#     for i, w in enumerate(weights):
#         cam += w * feature_map[0, i, :, :].detach().cpu().numpy()

#     cam = np.maximum(cam, 0)
#     cam = cv2.resize(cam, (112, 112))  # Match your frame size
#     cam = cam - np.min(cam)
#     cam = cam / np.max(cam)
#     return cam

def generate_cam(feature_map, gradients):
    weights = gradients.mean(dim=(2, 3))[0, :]  # Use .mean from PyTorch
    cam = torch.zeros(feature_map.shape[2:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * feature_map[0, i, :, :].cpu()

    cam = torch.relu(cam)
    cam = cam.numpy()
    cam = cv2.resize(cam, (112, 112))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam


from flask import send_from_directory

@app.route('/')
def homepage():
    return send_from_directory('frontend', 'index.html')

@app.route('/style.css')
def style():
    return send_from_directory('frontend', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('frontend', 'script.js')

@app.route('/Uploaded_Files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return send_from_directory('frontend', 'index.html')

    if request.method == 'POST':
        video = request.files['video']
        print(video.filename)

        # Save video
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

        # Get selected model from the form
        model_choice = request.form.get("model")

        # Detect
        prediction = detectFakeVideo(video_path, model_choice)

        # Interpret result
        output = "REAL" if prediction[0] == 1 else "FAKE"
        confidence = prediction[1]

        # Clean up
        os.remove(video_path)

        # Send response
        #data = {'output': output, 'confidence': confidence}
        ############# green
        cam_image_path = '/' + prediction[2] # Relative path to the output image
        data = {'output': output, 'confidence': confidence, 'cam_image': cam_image_path}
        ############# green !
        return jsonify(data)


app.run(port=3000);