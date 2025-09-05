from django.utils import timezone
import os
import sys
from django.conf import settings
from django.contrib import messages
from pydub import AudioSegment

# Add AI-Audio-Detector-main to sys.path for import
sys.path.append(os.path.join(settings.PROJECT_DIR, '..', '..', 'AI-Audio-Detector-main'))

# --- AUDIO DEEPFAKE DETECTION UTILITY & VIEW ---
def detect_audio_file(audio_file, request=None):
    """
    Utility to handle audio deepfake detection given a Django UploadedFile.
    Returns: result (dict), audio_file_url (for HTML audio player, absolute URL)
    """
    import importlib
    import tempfile
    import shutil
    from django.core.files.storage import default_storage
    from django.core.files.base import ContentFile
    from django.conf import settings
    try:
        audio_detector_module = importlib.import_module("main")
    except Exception as e:
        return {"label": "Error", "confidence": 0.0, "error": f"Failed to load audio detector: {e}"}, None

    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, audio_file.name)
    with open(temp_audio_path, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    # If file is not wav, convert to wav using pydub
    if not temp_audio_path.lower().endswith(".wav"):
        try:
            audio = AudioSegment.from_file(temp_audio_path)
            temp_wav_path = os.path.splitext(temp_audio_path)[0] + ".wav"
            audio.export(temp_wav_path, format="wav")
            audio_path_for_detection = temp_wav_path
        except Exception as e:
            shutil.rmtree(temp_dir)
            return {"label": "Error", "confidence": 0.0, "error": f"Audio conversion failed: {e}"}, None
    else:
        audio_path_for_detection = temp_audio_path

    # Run the deepfake audio detector
    try:
        if hasattr(audio_detector_module, "predict_audio_deepfake"):
            result = audio_detector_module.predict_audio_deepfake(audio_path_for_detection)
        else:
            if hasattr(audio_detector_module, "main"):
                result = audio_detector_module.main(audio_path_for_detection)
            else:
                result = "Audio detector function not found."
    except Exception as e:
        shutil.rmtree(temp_dir)
        return {"label": "Error", "confidence": 0.0, "error": f"Audio detection failed: {e}"}, None

    # Always return a dict: {'label':..., 'confidence':...}
    # If result is a dict and has 'label' and 'confidence', use as is.
    # If result is a dict with other keys, try to extract label/confidence.
    # If result is a string or other type, wrap as label.
    result_dict = None
    if isinstance(result, dict):
        # Try to extract 'label' and 'confidence'
        label = result.get('label')
        confidence = result.get('confidence')
        # If both present, use as is
        if label is not None and confidence is not None:
            result_dict = {'label': label, 'confidence': confidence}
        else:
            # Try to use first string key as label, and any float as confidence
            label_val = None
            conf_val = None
            for k, v in result.items():
                if label_val is None and isinstance(v, str):
                    label_val = v
                if conf_val is None and (isinstance(v, float) or isinstance(v, int)):
                    conf_val = float(v)
            if label_val is None:
                label_val = str(result)
            if conf_val is None:
                conf_val = 0.0
            result_dict = {'label': label_val, 'confidence': conf_val}
    else:
        # Not a dict: treat as label
        result_dict = {'label': str(result), 'confidence': 0.0}

    # Move the uploaded (possibly converted) wav file to a public directory for playback
    try:
        # Save the wav file to media/audio_uploads (create dir if needed)
        from django.conf import settings
        audio_uploads_dir = os.path.join(settings.MEDIA_ROOT, "audio_uploads")
        os.makedirs(audio_uploads_dir, exist_ok=True)
        audio_filename = os.path.basename(audio_path_for_detection)
        dest_audio_path = os.path.join(audio_uploads_dir, audio_filename)
        shutil.copy(audio_path_for_detection, dest_audio_path)
        rel_audio_url = os.path.join(settings.MEDIA_URL, "audio_uploads", audio_filename)
        # Convert to absolute URL if request is provided
        if request is not None:
            audio_file_url = request.build_absolute_uri(rel_audio_url)
        else:
            audio_file_url = rel_audio_url
    except Exception as e:
        shutil.rmtree(temp_dir)
        return {"label": "Error", "confidence": 0.0, "error": f"Audio file saving failed: {e}"}, None

    shutil.rmtree(temp_dir)
    return result_dict, audio_file_url

def audio_detection_view(request):
    """
    Standalone view for audio deepfake detection.
    """
    from django.shortcuts import render
    if request.method == "POST":
        audio_file = request.FILES.get("audio_file", None)
        if not audio_file:
            messages.error(request, "No audio file uploaded.")
            return render(request, "audio_detect.html")
        result, audio_file_url = detect_audio_file(audio_file)
        return render(request, "audio_detect.html", {"result": result, "audio_file_url": audio_file_url})
    else:
        return render(request, "audio_detect.html")
from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
import shutil
from PIL import Image as pImage
from django.conf import settings
from .forms import VideoUploadForm
import logging
logger = logging.getLogger(__name__)
from ultralytics import YOLO
from .forensics import fomm_likelihood, bg_face_flow_ratio, boundary_flicker, spectral_slope

# Audio extraction utility
def extract_audio_from_video(video_path, output_wav_path):
    """
    Extracts audio from a video file and saves it as a .wav file.
    Args:
        video_path (str): Path to the input video file.
        output_wav_path (str): Path where the extracted .wav file will be saved.
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            logger.warning(f"No audio stream found in video: {video_path}")
            return False
        clip.audio.write_audiofile(output_wav_path, codec='pcm_s16le')
        clip.close()
        return True
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return False

# Templates
index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

# Config
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
PREDICTION_THRESHOLD = 50.0  # configurable threshold 
# If your trained model used the mapping 0=REAL and 1=FAKE, set this to True to invert interpretation
INVERT_LABELS = True
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Global model loaded once
global_model = None
def load_global_model():
    global global_model
    if global_model is None:
        model = Model(2).to(device)
        model_name = 'model_97_acc_100_frames_FF_data.pt'
        model.load_state_dict(torch.load(os.path.join(settings.PROJECT_DIR, 'models', model_name), map_location=device))
        model.eval()
        global_model = model
        logger.info("Global model loaded")
    return global_model

# Global YOLO face detector
yolo_face_detector = YOLO(os.path.join(settings.PROJECT_DIR, "models", "yolov8n-face.pt"))

# Model definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional, batch_first=True)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
    # x shape should be (batch, seq, c, h, w)
        logger.info("Model.forward input shape: %s", tuple(x.shape))
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        logger.info("Feature map shape (fmap): %s", tuple(fmap.shape))
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        logger.info("After avgpool shape (for LSTM): %s", tuple(x.shape))
        x_lstm, _ = self.lstm(x, None)
        logger.info("LSTM output shape: %s", tuple(x_lstm.shape))
        out = self.dp(self.linear1(x_lstm[:, -1, :]))
        logger.info("Classifier out shape: %s", tuple(out.shape))
        return fmap, out

# Dataset class
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            results = yolo_face_detector(frame, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                continue
            try:
                box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                left, top, right, bottom = box[0], box[1], box[2], box[3]
                frame = frame[top:bottom, left:right, :]
            except Exception as e:
                logger.info(f"Face crop failed (YOLO): {e}")
                continue
            if i % a == first_frame:
                frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# Image utilities
def im_convert(tensor, video_file_name):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = image * [0.22803, 0.22145, 0.216989] + [0.43216, 0.394666, 0.37645]
    image = image * 255.0
    plt.imshow(image.astype('uint8'))
    plt.show()

# Prediction
def predict(model, img, path='./', video_file_name=""):
    with torch.inference_mode():
        fmap, logits = model(img.to(device))
    logger.info("Raw logits: %s", logits.detach().cpu().numpy())
    logits = sm(logits)
    logger.info("Softmax output: %s", logits.detach().cpu().numpy())
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    _, prediction = torch.max(logits, 1)
    pred_idx = int(prediction.item())
    # Log raw predicted index for debugging
    logger.info("Base model raw predicted index: %d", pred_idx)

    # Choose mapping based on INVERT_LABELS flag
    # Default assumption: 0 = FAKE, 1 = REAL
    if INVERT_LABELS:
        idx2label = {0: "REAL", 1: "FAKE"}
    else:
        idx2label = {0: "FAKE", 1: "REAL"}

    output_label = idx2label.get(pred_idx, "UNCERTAIN")

    confidence = logits[:, pred_idx].item() * 100
    logger.info("Base model predicted: %s (Confidence: %.2f%%)", output_label, confidence)

    # Return pred_idx as well so callers can inspect raw index if needed
    return [output_label, confidence, pred_idx]

# Heatmap
def plot_heat_map(i, model, img, path='./', video_file_name=''):
    fmap, logits = model(img.to(device))
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h * w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    result = heatmap * 0.5 + img * 0.8 * 255
    heatmap_name = video_file_name + "_heatmap_" + str(i) + ".png"
    image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
    cv2.imwrite(image_name, result)
    return image_name

# Model selection
def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass

    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        logger.info("No model found for the specified sequence length.")

    return final_model

# Allowed files
ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv', 'mov'])

def allowed_video_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# Index view
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        for key in ['file_name', 'preprocessed_images', 'faces_cropped_images']:
            request.session.pop(key, None)
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})
            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            if not allowed_video_file(video_file.name):
                video_upload_form.add_error("upload_video_file", "Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            saved_video_file = 'uploaded_file_' + str(int(time.time())) + "." + video_file_ext
            save_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            with open(save_path, 'wb') as vFile:
                shutil.copyfileobj(video_file, vFile)
            request.session['file_name'] = save_path
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

# Predict page
def predict_page(request):
    """
    Unified predict page: handles video detection result (always), and audio detection if audio POSTed.
    """
    audio_result = None
    audio_file_url = None
    # Handle audio POST if present
    if request.method == "POST" and request.FILES.get("audio_file", None):
        audio_file = request.FILES["audio_file"]
        audio_result, audio_file_url = detect_audio_file(audio_file, request=request)

    # Video detection logic (always runs for GET and POST)
    if 'file_name' not in request.session:
        return redirect("ml_app:index")
    video_file = request.session['file_name']
    sequence_length = request.session['sequence_length']
    path_to_videos = [video_file]
    video_filename = os.path.basename(video_file)
    production_video_name = video_filename if settings.DEBUG else os.path.join('/home/app/staticfiles/', video_filename)

    video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)
    model = load_global_model()
    start_time = time.time()
    preprocessed_images = []
    faces_cropped_images = []
    padding = 60
    faces_found = 0
    scores = {"bg_flow_ratio": [], "boundary_flicker": [], "spec_slope": []}
    prev_frame = None
    embeddings = []
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    video_file_name_only = os.path.splitext(video_filename)[0]
    while cap.isOpened() and frame_count < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_name = f"{video_file_name_only}_preprocessed_{frame_count}.png"
        pImage.fromarray(rgb_frame, 'RGB').save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name))
        preprocessed_images.append(image_name)
        results = yolo_face_detector(frame, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            continue
        try:
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            h_frame, w_frame = frame.shape[:2]
            pad_left = max(0, left - padding)
            pad_top = max(0, top - padding)
            pad_right = min(w_frame, right + padding)
            pad_bottom = min(h_frame, bottom + padding)
            frame_face = frame[pad_top:pad_bottom, pad_left:pad_right]
            if frame_face is None or frame_face.size == 0:
                continue
        except Exception as e:
            logger.info(f"Face crop failed (YOLO): {e}")
            continue
        rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
        image_name = f"{video_file_name_only}_cropped_faces_{frame_count}.png"
        pImage.fromarray(rgb_face, 'RGB').save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name))
        faces_found += 1
        faces_cropped_images.append(image_name)
        if prev_frame is not None:
            try:
                scores["bg_flow_ratio"].append(bg_face_flow_ratio(prev_frame, frame, box))
                scores["boundary_flicker"].append(boundary_flicker(prev_frame, frame, box))
            except Exception as e:
                logger.info(f"FOMM metric failed: {e}")
        try:
            face_gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
            scores["spec_slope"].append(spectral_slope(cv2.resize(face_gray, (112,112))))
        except Exception as e:
            logger.info(f"Spectral slope failed: {e}")
        # Compute aligned face embeddings
        try:
            rgb_face_for_enc = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb_face_for_enc)
            if landmarks_list:
                landmarks = landmarks_list[0]
                if "left_eye" in landmarks and "right_eye" in landmarks:
                    left_eye = np.mean(landmarks["left_eye"], axis=0)
                    right_eye = np.mean(landmarks["right_eye"], axis=0)
                    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
                    angle = np.degrees(np.arctan2(dy, dx))
                    h, w = rgb_face_for_enc.shape[:2]
                    center = tuple(np.mean([left_eye, right_eye], axis=0))
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aligned_face = cv2.warpAffine(rgb_face_for_enc, M, (w, h))
                    encs = face_recognition.face_encodings(aligned_face)
                else:
                    encs = face_recognition.face_encodings(rgb_face_for_enc)
            else:
                encs = face_recognition.face_encodings(rgb_face_for_enc)
            if encs:
                embeddings.append(encs[0])
        except Exception as e:
            logger.info(f"Embedding failed: {e}")
        prev_frame = frame
    cap.release()
    logger.info("<=== | Videos Splitting and Face Cropping Done | ===>")
    logger.info("--- %s seconds ---", time.time() - start_time)
    if faces_found == 0:
        return render(request, predict_template_name, {"no_faces": True})
    logger.debug("FOMM raw arrays: flow=%s, flick=%s, spec=%s",
                 str(scores.get('bg_flow_ratio', []))[:200],
                 str(scores.get('boundary_flicker', []))[:200],
                 str(scores.get('spec_slope', []))[:200])
    try:
        heatmap_images = []
        output = ""
        confidence = 0.0
        for i in range(len(path_to_videos)):
            logger.info("<=== | Started Prediction | ===>")
            prediction = predict(model, video_dataset[i], './', video_file_name_only)
            confidence = round(prediction[1], 1)
            output = prediction[0]
            base_pred_idx = prediction[2]
            logger.info("Base model index returned: %d", base_pred_idx)
            logger.info("Prediction: %s Confidence: %s", output, confidence)
            logger.info("<=== | Prediction Done | ===>")
            logger.info("--- %s seconds ---", time.time() - start_time)
        fomm_score = fomm_likelihood(scores)
        from .forensics import identity_variance
        identity_score = identity_variance(embeddings)
        logger.info("Identity variance score: %.2f", identity_score)
        logger.info("FOMM counts: flow=%d, flick=%d, spec=%d",
                    len(scores.get('bg_flow_ratio', [])),
                    len(scores.get('boundary_flicker', [])),
                    len(scores.get('spec_slope', [])))
        final_confidence = confidence
        logger.info("Final decision (base model only): %s (Confidence=%.2f)", output, final_confidence)
        context = {
            'preprocessed_images': preprocessed_images,
            'faces_cropped_images': faces_cropped_images,
            'heatmap_images': heatmap_images if heatmap_images else None,
            'original_video': production_video_name,
            'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
            'output': output,
            'confidence': final_confidence,
            'fomm_score': fomm_score,
            'identity_score': identity_score,
            'base_pred_idx': base_pred_idx,
            # Audio results for predict.html (if present)
            'result': audio_result,
            'audio_file_url': audio_file_url
        }
        return render(request, predict_template_name, context)
    except Exception as e:
        logger.error("Exception occurred during prediction: %s", e)
        return render(request, 'cuda_full.html')

# Static pages
def about(request):
    return render(request, about_template_name)

def handler404(request, exception):
    return render(request, '404.html', status=404)

def cuda_full(request):
    return render(request, 'cuda_full.html')

# New view: video_upload_page
from django.views.decorators.csrf import csrf_exempt

import os
import sys
import time
import shutil
import tempfile
import logging
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch import nn
import cv2
from PIL import Image as pImage
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .forms import VideoUploadForm
from ultralytics import YOLO
import face_recognition
from .forensics import fomm_likelihood, bg_face_flow_ratio, boundary_flicker, spectral_slope

logger = logging.getLogger(__name__)

# --- AUDIO DEEPFAKE DETECTION UTILITY & VIEW ---
def detect_audio_file(audio_file, request=None):
    """
    Utility to handle audio deepfake detection given a Django UploadedFile.
    Returns: result (dict), audio_file_url (for HTML audio player, absolute URL)
    """
    import importlib
    import tempfile
    import shutil
    from pydub import AudioSegment
    try:
        # Add AI-Audio-Detector-main to sys.path for import
        sys.path.append(os.path.join(settings.PROJECT_DIR, '..', '..', 'AI-Audio-Detector-main'))
        audio_detector_module = importlib.import_module("main")
    except Exception as e:
        return {"label": "Error", "confidence": 0.0, "error": f"Failed to load audio detector: {e}"}, None

    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, audio_file.name)
    with open(temp_audio_path, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    # If file is not wav, convert to wav using pydub
    if not temp_audio_path.lower().endswith(".wav"):
        try:
            audio = AudioSegment.from_file(temp_audio_path)
            temp_wav_path = os.path.splitext(temp_audio_path)[0] + ".wav"
            audio.export(temp_wav_path, format="wav")
            audio_path_for_detection = temp_wav_path
        except Exception as e:
            shutil.rmtree(temp_dir)
            return {"label": "Error", "confidence": 0.0, "error": f"Audio conversion failed: {e}"}, None
    else:
        audio_path_for_detection = temp_audio_path

    # Run the deepfake audio detector
    try:
        if hasattr(audio_detector_module, "predict_audio_deepfake"):
            result = audio_detector_module.predict_audio_deepfake(audio_path_for_detection)
        else:
            if hasattr(audio_detector_module, "main"):
                result = audio_detector_module.main(audio_path_for_detection)
            else:
                result = "Audio detector function not found."
    except Exception as e:
        shutil.rmtree(temp_dir)
        return {"label": "Error", "confidence": 0.0, "error": f"Audio detection failed: {e}"}, None

    # Always return a dict: {'label':..., 'confidence':...}
    result_dict = None
    if isinstance(result, dict):
        label = result.get('label')
        confidence = result.get('confidence')
        if label is not None and confidence is not None:
            result_dict = {'label': label, 'confidence': confidence}
        else:
            label_val = None
            conf_val = None
            for k, v in result.items():
                if label_val is None and isinstance(v, str):
                    label_val = v
                if conf_val is None and (isinstance(v, float) or isinstance(v, int)):
                    conf_val = float(v)
            if label_val is None:
                label_val = str(result)
            if conf_val is None:
                conf_val = 0.0
            result_dict = {'label': label_val, 'confidence': conf_val}
    else:
        result_dict = {'label': str(result), 'confidence': 0.0}

    # Move the uploaded (possibly converted) wav file to a public directory for playback
    try:
        audio_uploads_dir = os.path.join(settings.MEDIA_ROOT, "audio_uploads")
        os.makedirs(audio_uploads_dir, exist_ok=True)
        audio_filename = os.path.basename(audio_path_for_detection)
        dest_audio_path = os.path.join(audio_uploads_dir, audio_filename)
        shutil.copy(audio_path_for_detection, dest_audio_path)
        rel_audio_url = os.path.join(settings.MEDIA_URL, "audio_uploads", audio_filename)
        if request is not None:
            audio_file_url = request.build_absolute_uri(rel_audio_url)
        else:
            audio_file_url = rel_audio_url
    except Exception as e:
        shutil.rmtree(temp_dir)
        return {"label": "Error", "confidence": 0.0, "error": f"Audio file saving failed: {e}"}, None

    shutil.rmtree(temp_dir)
    return result_dict, audio_file_url

def audio_detection_view(request):
    """
    Standalone view for audio deepfake detection.
    """
    if request.method == "POST":
        audio_file = request.FILES.get("audio_file", None)
        if not audio_file:
            messages.error(request, "No audio file uploaded.")
            return render(request, "audio_detect.html")
        result, audio_file_url = detect_audio_file(audio_file, request=request)
        return render(request, "audio_detect.html", {"result": result, "audio_file_url": audio_file_url})
    else:
        return render(request, "audio_detect.html")

# --- AUDIO EXTRACTION UTILITY ---
def extract_audio_from_video(video_path, output_wav_path):
    """
    Extracts audio from a video file and saves it as a .wav file.
    Args:
        video_path (str): Path to the input video file.
        output_wav_path (str): Path where the extracted .wav file will be saved.
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            logger.warning(f"No audio stream found in video: {video_path}")
            return False
        clip.audio.write_audiofile(output_wav_path, codec='pcm_s16le')
        clip.close()
        return True
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return False

# --- VIDEO MODEL, DATASET, AND UTILITIES ---
index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
PREDICTION_THRESHOLD = 50.0
INVERT_LABELS = True
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

global_model = None
def load_global_model():
    global global_model
    if global_model is None:
        model = Model(2).to(device)
        model_name = 'model_97_acc_100_frames_FF_data.pt'
        model.load_state_dict(torch.load(os.path.join(settings.PROJECT_DIR, 'models', model_name), map_location=device))
        model.eval()
        global_model = model
        logger.info("Global model loaded")
    return global_model

yolo_face_detector = YOLO(os.path.join(settings.PROJECT_DIR, "models", "yolov8n-face.pt"))

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional, batch_first=True)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        out = self.dp(self.linear1(x_lstm[:, -1, :]))
        return fmap, out

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            results = yolo_face_detector(frame, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                continue
            try:
                box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                left, top, right, bottom = box[0], box[1], box[2], box[3]
                frame = frame[top:bottom, left:right, :]
            except Exception as e:
                logger.info(f"Face crop failed (YOLO): {e}")
                continue
            if i % a == first_frame:
                frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def im_convert(tensor, video_file_name):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img, path='./', video_file_name=""):
    with torch.inference_mode():
        fmap, logits = model(img.to(device))
    logits = sm(logits)
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    _, prediction = torch.max(logits, 1)
    pred_idx = int(prediction.item())
    if INVERT_LABELS:
        idx2label = {0: "REAL", 1: "FAKE"}
    else:
        idx2label = {0: "FAKE", 1: "REAL"}
    output_label = idx2label.get(pred_idx, "UNCERTAIN")
    confidence = logits[:, pred_idx].item() * 100
    return [output_label, confidence, pred_idx]

def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    import glob
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))
    for model_path in list_models:
        model_name.append(os.path.basename(model_path))
    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass
    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        logger.info("No model found for the specified sequence length.")
    return final_model

ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv', 'mov'])
def allowed_video_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        for key in ['file_name', 'preprocessed_images', 'faces_cropped_images']:
            request.session.pop(key, None)
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})
            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            if not allowed_video_file(video_file.name):
                video_upload_form.add_error("upload_video_file", "Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            saved_video_file = 'uploaded_file_' + str(int(time.time())) + "." + video_file_ext
            save_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            with open(save_path, 'wb') as vFile:
                shutil.copyfileobj(video_file, vFile)
            request.session['file_name'] = save_path
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    """
    Unified predict page: handles video detection result (always), and audio detection if audio POSTed.
    """
    audio_result = None
    audio_file_url = None
    if request.method == "POST" and request.FILES.get("audio_file", None):
        audio_file = request.FILES["audio_file"]
        audio_result, audio_file_url = detect_audio_file(audio_file, request=request)
    if 'file_name' not in request.session:
        return redirect("ml_app:index")
    video_file = request.session['file_name']
    sequence_length = request.session['sequence_length']
    path_to_videos = [video_file]
    video_filename = os.path.basename(video_file)
    production_video_name = video_filename if settings.DEBUG else os.path.join('/home/app/staticfiles/', video_filename)
    video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)
    model = load_global_model()
    start_time = time.time()
    preprocessed_images = []
    faces_cropped_images = []
    padding = 60
    faces_found = 0
    scores = {"bg_flow_ratio": [], "boundary_flicker": [], "spec_slope": []}
    prev_frame = None
    embeddings = []
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    video_file_name_only = os.path.splitext(video_filename)[0]
    while cap.isOpened() and frame_count < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_name = f"{video_file_name_only}_preprocessed_{frame_count}.png"
        pImage.fromarray(rgb_frame, 'RGB').save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name))
        preprocessed_images.append(image_name)
        results = yolo_face_detector(frame, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            continue
        try:
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            h_frame, w_frame = frame.shape[:2]
            pad_left = max(0, left - padding)
            pad_top = max(0, top - padding)
            pad_right = min(w_frame, right + padding)
            pad_bottom = min(h_frame, bottom + padding)
            frame_face = frame[pad_top:pad_bottom, pad_left:pad_right]
            if frame_face is None or frame_face.size == 0:
                continue
        except Exception as e:
            logger.info(f"Face crop failed (YOLO): {e}")
            continue
        rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
        image_name = f"{video_file_name_only}_cropped_faces_{frame_count}.png"
        pImage.fromarray(rgb_face, 'RGB').save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name))
        faces_found += 1
        faces_cropped_images.append(image_name)
        if prev_frame is not None:
            try:
                scores["bg_flow_ratio"].append(bg_face_flow_ratio(prev_frame, frame, box))
                scores["boundary_flicker"].append(boundary_flicker(prev_frame, frame, box))
            except Exception as e:
                logger.info(f"FOMM metric failed: {e}")
        try:
            face_gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
            scores["spec_slope"].append(spectral_slope(cv2.resize(face_gray, (112,112))))
        except Exception as e:
            logger.info(f"Spectral slope failed: {e}")
        try:
            rgb_face_for_enc = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb_face_for_enc)
            if landmarks_list:
                landmarks = landmarks_list[0]
                if "left_eye" in landmarks and "right_eye" in landmarks:
                    left_eye = np.mean(landmarks["left_eye"], axis=0)
                    right_eye = np.mean(landmarks["right_eye"], axis=0)
                    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
                    angle = np.degrees(np.arctan2(dy, dx))
                    h, w = rgb_face_for_enc.shape[:2]
                    center = tuple(np.mean([left_eye, right_eye], axis=0))
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aligned_face = cv2.warpAffine(rgb_face_for_enc, M, (w, h))
                    encs = face_recognition.face_encodings(aligned_face)
                else:
                    encs = face_recognition.face_encodings(rgb_face_for_enc)
            else:
                encs = face_recognition.face_encodings(rgb_face_for_enc)
            if encs:
                embeddings.append(encs[0])
        except Exception as e:
            logger.info(f"Embedding failed: {e}")
        prev_frame = frame
    cap.release()
    if faces_found == 0:
        return render(request, predict_template_name, {"no_faces": True})
    try:
        heatmap_images = []
        output = ""
        confidence = 0.0
        for i in range(len(path_to_videos)):
            prediction = predict(model, video_dataset[i], './', video_file_name_only)
            confidence = round(prediction[1], 1)
            output = prediction[0]
            base_pred_idx = prediction[2]
        fomm_score = fomm_likelihood(scores)
        from .forensics import identity_variance
        identity_score = identity_variance(embeddings)
        final_confidence = confidence
        context = {
            'preprocessed_images': preprocessed_images,
            'faces_cropped_images': faces_cropped_images,
            'heatmap_images': heatmap_images if heatmap_images else None,
            'original_video': production_video_name,
            'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
            'output': output,
            'confidence': final_confidence,
            'fomm_score': fomm_score,
            'identity_score': identity_score,
            'base_pred_idx': base_pred_idx,
            'result': audio_result,
            'audio_file_url': audio_file_url
        }
        return render(request, predict_template_name, context)
    except Exception as e:
        logger.error("Exception occurred during prediction: %s", e)
        return render(request, 'cuda_full.html')

def about(request):
    return render(request, about_template_name)

def handler404(request, exception):
    return render(request, '404.html', status=404)

def cuda_full(request):
    return render(request, 'cuda_full.html')

from django.conf import settings
import os

def video_upload_page(request):
    """
    Handles the 'Video' button from index.html.
    - GET: Render video_upload.html.
    - POST: Handle video upload and sequence length.
    """
    if request.method == "GET":
        form = VideoUploadForm()
        return render(request, "video_upload.html", {"form": form})
    elif request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data["upload_video_file"]
            sequence_length = form.cleaned_data["sequence_length"]
            # Validate file extension
            if not allowed_video_file(video_file.name):
                form.add_error("upload_video_file", "Only video files are allowed")
                return render(request, "video_upload.html", {"form": form})
            # Validate sequence length
            if sequence_length <= 0:
                form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, "video_upload.html", {"form": form})
            # Validate file size
            if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                form.add_error("upload_video_file", "Maximum file size 100 MB")
                return render(request, "video_upload.html", {"form": form})
            # Save with timestamped filename in uploaded_videos/
            import time
            video_file_ext = video_file.name.split('.')[-1]
            ts = int(time.time())
            saved_video_file = f"uploaded_videos_{ts}.{video_file_ext}"
            save_dir = os.path.join(settings.PROJECT_DIR, "uploaded_videos")
            os.makedirs(save_dir, exist_ok=True)
            video_save_path = os.path.join(save_dir, saved_video_file)
            with open(video_save_path, "wb+") as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            # Store file path and sequence_length in session
            request.session['file_name'] = video_save_path
            request.session['sequence_length'] = sequence_length
            # Redirect to predict page
            return redirect('ml_app:predict')
        else:
            return render(request, "video_upload.html", {"form": form})
from django.shortcuts import render

def audio_upload_view(request):
    """
    Handles audio upload and detection from the template.
    """
    from django.contrib import messages

    context = {}

    if request.method == "POST":
        audio_file = request.FILES.get("audio_file", None)
        if not audio_file:
            messages.error(request, "No audio file uploaded.")
            return render(request, "audio_detect.html", context)

        # Run the detection using your utility
        result, audio_file_url = detect_audio_file(audio_file, request=request)
        context['result'] = result
        context['audio_file_url'] = audio_file_url

    return render(request, "audio_detect.html", context)