Deepfake Video Detection using Deep Learning (ResNeXt + LSTM)
📌 Overview
This project implements a deepfake video detection system using a hybrid approach of ResNeXt (CNN) for feature extraction and LSTM (RNN) for temporal sequence modeling.
It allows detection of manipulated (deepfake) videos by analyzing sequences of frames and classifying them as real or fake.

Maintained by Anshuman Mishra.
⚙️ Features
• Deep Learning Models: ResNeXt + LSTM
• Transfer Learning: pretrained ResNeXt backbone for feature extraction
• Temporal Analysis: LSTM layers capture frame dependencies
• Web Interface: Django application for uploading and testing videos
• Dockerized Deployment: spin up containers in seconds
• Cross-Platform: works on CUDA (NVIDIA GPUs) and CPU-only machines
• Git LFS: supports large files (models, videos)
📂 Project Structure
Video_Deepfake_detection_using_deep_learning
│
├── Django Application/     # Web interface for uploading and testing videos
├── Model Creation/         # Training pipeline and scripts for ResNeXt + LSTM
├── Documentation/          # Detailed docs and notes
├── github_assets/          # Demo images and GIFs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation (this file)

🚀 Installation & Setup
1. Clone the Repository:
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

2. Create Virtual Environment:
   python3 -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows

3. Install Dependencies:
   pip install -r requirements.txt

4. Run the Django Application:
   cd "Django Application"
   python manage.py runserver

Visit the app at: http://127.0.0.1:8000/
📊 Results
Model	Frames Used	Accuracy
ResNeXt + LSTM	10	~84%
ResNeXt + LSTM	20	~88%
ResNeXt + LSTM	40	~89%
ResNeXt + LSTM	60	~90%
ResNeXt + LSTM	100	~93%
🖼️ Demo
System Architecture and detection example are available in github_assets/ folder.
🔮 Future Improvements
• Batch processing for entire videos
• Cloud deployment (Heroku, Render, AWS)
• Open-source API for deepfake detection
• Faster inference optimizations
• Improved datasets and architectures
🤝 Contribution
Contributions, improvements, and ideas are welcome!
Feel free to fork the repo, open issues, or submit pull requests.

Maintained by: Anshuman Mishra
📜 License
Licensed under the GPL v3 License.
See LICENSE for details.