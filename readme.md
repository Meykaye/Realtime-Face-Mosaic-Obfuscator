
# Realtime Face Mosaic Obfuscator

A Python-based real-time face anonymization tool using OpenCV and a deep learning face detector. This project detects human faces from a webcam stream and applies a pixelated mosaic blur to anonymize them effectively.

---

## 📌 Features

- 🎥 Real-time webcam face detection
- 🧠 Uses a deep learning SSD face detector (`res10_300x300_ssd_iter_140000.caffemodel`)
- 🧊 Mosaic-style pixelated blur for anonymization
- 💻 Adjustable pixelation scale
- 🪟 Enlarged display window
- ⌨️ Exit on any key press
---

## 🛠️ Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:

```py
pip install opencv-python numpy
````

---

## 🧪 How It Works

1. Captures video stream from your webcam.
2. Detects faces using a deep learning SSD-based face detector.
3. Applies a mosaic blur to the region of each detected face.
4. Displays the output in a resizable window.
5. Press any key to exit the application.

---

## 🧾 Files

* `main.py` - Main script for face mosaic obfuscation
* `deploy.prototxt` - Configuration file for the SSD face detector
* `res10_300x300_ssd_iter_140000.caffemodel` - Pre-trained model weights

---

## 🧠 Model Info

The face detection model used here is based on a Single Shot Multibox Detector (SSD) with a ResNet-10 architecture. It is pre-trained on a face dataset and optimized for speed and accuracy.

---

## 🖼️ Example Output

| Original Video Feed | Mosaic Applied |
| ------------------- | -------------- |
| Face visible        | Face pixelated |

---

## 📤 Future Improvements

* Add command-line arguments for tuning mosaic scale
* Enable video recording and saving output
* Allow selection of input sources (e.g., video files)

---

## 📄 License

This project is licensed under the MIT License.


