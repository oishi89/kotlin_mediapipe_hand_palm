# Palm Detection - MediaPipe SSD Android Demo

This is a **lightweight and minimal sample project** that demonstrates how to extract bounding boxes from MediaPipe’s `palm_detection_lite.tflite` SSD model on Android using Kotlin and TensorFlow Lite.

## ✨ Features

✅ Matches MediaPipe’s official anchor generation  
✅ Decodes bounding boxes using predefined priors  
✅ Applies sigmoid to compute meaningful confidence scores  
✅ Filters boxes by confidence threshold (e.g. `score > 0.5`)  
✅ Performs **Non-Max Suppression (NMS)** to remove overlaps  
✅ Returns the **best palm bounding box and 7 keypoints** (this demo focuses only on the **bounding box**)
