# LogoClassification
Dự án này chỉ ra cách đào tạo và dự đoán 3 lớp logo với keras và tensorflow lite

## Dependencies
python3, keras 2.2.4, tensorflow 1.15.0

## Train your images
Link dataset
```bash
python3 train.py
```
## Convert your model to tflite
```bash
python3 convert2tflite.py
```

## Inference with .h5
```bash
python3 predict.py
```

## Inference with .tflite
```bash
python3 inference_on_tflite.py
```
## Application example
[![Product Sorting System with Tensorflow Lite, Raspberry Pi and Arduino Demo
]
