# Palm Detection - MediaPipe SSD Android Demo

This is a **lightweight and minimal sample project** that demonstrates how to extract bounding boxes from MediaPipe’s `palm_detection_lite.tflite` SSD model on Android using Kotlin and TensorFlow Lite.

## Features

1. Matches MediaPipe’s official anchor generation  
2. Decodes bounding boxes using predefined priors  
3. Applies sigmoid to compute meaningful confidence scores  
4. Filters boxes by confidence threshold (e.g. `score > 0.5`)  
5. Performs **Non-Max Suppression (NMS)** to remove overlaps  
6. Returns the **best palm bounding box and 7 keypoints** (this demo focuses only on the **bounding box**)
