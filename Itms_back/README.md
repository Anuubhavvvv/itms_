# Vehicle ANPR Pipeline

A complete pipeline for vehicle detection, number plate detection, and OCR text extraction using YOLOv8, ANPR models, and PaddleOCR.

## Features

- **Vehicle Detection**: Detect vehicles in images/videos using YOLOv8
- **Number Plate Detection**: Detect number plates using custom ANPR model
- **OCR Processing**: Extract text from number plates using PaddleOCR
- **Batch Processing**: Process multiple images or video files
- **Result Visualization**: Save annotated images with detection results
- **Performance Metrics**: Track processing times for each stage

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the model files in the correct locations:
   - `models/vehicle_yolov8.pt` - YOLOv8 vehicle detection model
   - `models/anpr_best.pt` - ANPR number plate detection model

## Quick Start

### Basic Usage

```python
from vehicle_anpr_pipeline import VehicleANPRPipeline

# Initialize the pipeline
pipeline = VehicleANPRPipeline(
    vehicle_confidence=0.5,    # Vehicle detection confidence threshold
    plate_confidence=0.5,      # Plate detection confidence threshold
    ocr_language='en'          # OCR language
)

# Process a single image
results = pipeline.process_image("path/to/image.jpg", save_results=True)

# Print results
print(f"Vehicles detected: {len(results['vehicles'])}")
print(f"Number plates detected: {len(results['number_plates'])}")
for i, ocr_result in enumerate(results['ocr_results']):
    if ocr_result['detected_text']:
        print(f"Plate {i+1}: {', '.join(ocr_result['detected_text'])}")
```

### Process Video

```python
# Process a video file
results_list = pipeline.process_video(
    "input_video.mp4",
    output_path="output_video.mp4",
    frame_skip=5  # Process every 5th frame
)
```

## Pipeline Components

### 1. Vehicle Detection (`processors/vehicle_detect.py`)
- Uses YOLOv8 model for vehicle detection
- Returns bounding boxes and confidence scores
- Supports cropping of detected vehicle regions

### 2. ANPR Detection (`processors/anpr_detect.py`)
- Uses custom ANPR model for number plate detection
- Detects plates within vehicle regions
- Returns plate bounding boxes and confidence scores

### 3. OCR Processing (`processors/ocr_method.py`)
- Uses PaddleOCR for text extraction
- Supports multiple languages (English, Chinese, French, German, Korean, Japanese)
- Includes image preprocessing for better OCR results

## Configuration

### Pipeline Parameters

- `vehicle_confidence`: Minimum confidence for vehicle detection (default: 0.5)
- `plate_confidence`: Minimum confidence for plate detection (default: 0.5)
- `ocr_language`: Language for OCR processing (default: 'en')

### Supported OCR Languages

- `en`: English
- `ch`: Chinese
- `fr`: French
- `german`: German
- `korean`: Korean
- `japan`: Japanese

## Output Structure

The pipeline returns a dictionary with the following structure:

```python
{
    'input_image': numpy_array,           # Original input image
    'processing_time': {                  # Processing times for each stage
        'vehicle_detection': float,
        'plate_detection': float,
        'ocr_processing': float,
        'total': float
    },
    'vehicles': [                        # Vehicle detection results
        {
            'bbox': [x1, y1, x2, y2],
            'confidence': float,
            'class_id': int,
            'class_name': str
        }
    ],
    'number_plates': [                   # Number plate detection results
        {
            'bbox': [x1, y1, x2, y2],
            'confidence': float,
            'class_id': int,
            'class_name': str,
            'vehicle_index': int,
            'vehicle_bbox': [x1, y1, x2, y2]
        }
    ],
    'ocr_results': [                     # OCR extraction results
        {
            'detected_text': [str],
            'confidence_scores': [float],
            'bounding_boxes': [list],
            'plate_bbox': [x1, y1, x2, y2],
            'detection_confidence': float,
            'plate_index': int
        }
    ]
}
```

## File Structure

```
C:\Itms_back\
├── models\
│   ├── vehicle_yolov8.pt      # YOLOv8 vehicle detection model
│   └── anpr_best.pt           # ANPR number plate detection model
├── processors\
│   ├── vehicle_detect.py      # Vehicle detection module
│   ├── anpr_detect.py         # ANPR detection module
│   └── ocr_method.py          # OCR processing module
├── vehicle_anpr_pipeline.py  # Main pipeline class
├── example_usage.py          # Usage examples
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Examples

### Example 1: Process Single Image

```python
from vehicle_anpr_pipeline import VehicleANPRPipeline

pipeline = VehicleANPRPipeline()
results = pipeline.process_image("car_image.jpg", save_results=True)

# Results will be saved to "output/" directory:
# - original.jpg: Original image
# - annotated.jpg: Image with all detections
# - plate_1.jpg, plate_2.jpg: Cropped number plates
# - ocr_results.txt: Text extraction results
```

### Example 2: Process Video

```python
# Process video with annotations
results_list = pipeline.process_video(
    "traffic_video.mp4",
    output_path="annotated_video.mp4",
    frame_skip=10  # Process every 10th frame
)
```

### Example 3: Batch Processing

```python
import os

# Process all images in a directory
input_dir = "input_images"
for image_file in os.listdir(input_dir):
    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_dir, image_file)
        results = pipeline.process_image(image_path, save_results=True)
        print(f"Processed {image_file}: {len(results['vehicles'])} vehicles, {len(results['number_plates'])} plates")
```

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster processing
2. **Frame Skipping**: For video processing, use `frame_skip` parameter to process every nth frame
3. **Confidence Thresholds**: Adjust confidence thresholds based on your use case
4. **Image Preprocessing**: The OCR module includes automatic image preprocessing for better results

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model files are in the correct paths
2. **CUDA Errors**: Install appropriate PyTorch version for your system
3. **OCR Language Issues**: Make sure the specified language is supported
4. **Memory Issues**: Reduce batch size or use smaller images

### Dependencies

- Python 3.7+
- PyTorch 1.9+
- OpenCV 4.5+
- PaddleOCR 2.6+
- Ultralytics 8.0+

## License

This project is for educational and research purposes. Please ensure you have appropriate licenses for the models and datasets used.
