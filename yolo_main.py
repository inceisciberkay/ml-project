from ultralytics import YOLO
import os
import yaml
from evaluation_utils import plot_confusion_matrix, calculate_evaluation_metrics

# # train detectors
# model = YOLO("yolo_config/yolov8n.pt")
# model.train(data="yolo_config/yolo_cfg.yaml", epochs=15, imgsz=224, batch=4)
# model.export()

# model = YOLO("yolo_config/yolov8n.pt")
# model.train(data="yolo_config/yolo_cfg.yaml", epochs=15, imgsz=224, batch=8)
# model.export()

# model = YOLO("yolo_config/yolov8n.pt")
# model.train(data="yolo_config/yolo_cfg.yaml", epochs=15, imgsz=224, batch=16)
# model.export()

# model = YOLO("yolo_config/yolov8n.pt")
# model.train(data="yolo_config/yolo_cfg.yaml", epochs=15, imgsz=224, batch=32)
# model.export()

# model = YOLO("yolo_config/yolov8n.pt")
# model.train(data="yolo_config/yolo_cfg.yaml", epochs=20, imgsz=224)
# model.export()

# model = YOLO("yolo_config/yolov8n.pt")
# model.train(data="yolo_config/yolo_cfg.yaml", epochs=30, imgsz=224)
# model.export()

# model = YOLO("yolo_config/yolov8n.pt")
# model.train(data="yolo_config/yolo_cfg.yaml", epochs=50, imgsz=224)
# model.export()

# # train classifiers
# model = YOLO("yolo_config/yolov8n-cls.pt")
# model.train(data="dataset/images", epochs=15, batch=4)
# model.export()

# model = YOLO("yolo_config/yolov8n-cls.pt")
# model.train(data="dataset/images", epochs=15, batch=8)
# model.export()

# model = YOLO("yolo_config/yolov8n-cls.pt")
# model.train(data="dataset/images", epochs=15, batch=16)
# model.export()

# model = YOLO("yolo_config/yolov8n-cls.pt")
# model.train(data="dataset/images", epochs=15, batch=32)
# model.export()

# model = YOLO("yolo_config/yolov8n-cls.pt")
# model.train(data="dataset/images", epochs=20)
# model.export()

# model = YOLO("yolo_config/yolov8n-cls.pt")
# model.train(data="dataset/images", epochs=30)
# model.export()

# model = YOLO("yolo_config/yolov8n-cls.pt")
# model.train(data="dataset/images", epochs=50)
# model.export()

def evaluate_detect_model(run_path, split):
    config_path = os.path.join(run_path, 'args.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    imgsz = config['imgsz']
    model_path = os.path.join(run_path, 'weights/best.torchscript')
    model = YOLO(model_path)  # load a custom model
    FN = 0
    TP = 0
    for root, dirs, files in os.walk(f'dataset/images/{split}/fractured'):
        for file in files:
            file_path = os.path.join(root, file)
            results = model(source=file_path, imgsz=imgsz, conf=0.15)  # no arguments needed, dataset and settings remembered
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                if boxes.data.numel() == 0:
                    FN += 1
                else:
                    TP += 1
    FP = 0
    TN = 0
    for root, dirs, files in os.walk(f'dataset/images/{split}/nonfractured'):
        for file in files:
            file_path = os.path.join(root, file)
            results = model(source=file_path, imgsz=imgsz, conf=0.15)  # no arguments needed, dataset and settings remembered
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                if boxes.data.numel() > 0:
                    FP += 1
                else:
                    TN += 1

    return TP, FP, FN, TN

detect_runs = [
    'runs/detect/train',
    'runs/detect/train2',
    'runs/detect/train3',
    'runs/detect/train4',
    'runs/detect/train5',
    'runs/detect/train7',
    'runs/detect/train8',
    'runs/detect/train9',
    'runs/detect/train10',
    'runs/detect/train11',
    'runs/detect/train12',
]

classify_runs = [
    'runs/classify/train',
    'runs/classify/train2',
    'runs/classify/train3',
    'runs/classify/train4',
    'runs/classify/train5',
    'runs/classify/train6'
]

# Get validation results of detect runs
for run in detect_runs:
    TP, FP, FN, TN = evaluate_detect_model(run, 'validation')
    accuracy, precision, recall, f1_score, f2_score = calculate_evaluation_metrics(TP, FP, FN, TN)
    plot_confusion_matrix(TP, FP, FN, TN)
    with open('yolo_results.txt', "a+") as file:
        result = f"""\
Model run path: {run}
TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1 Score: {f1_score}
F2 Score: {f2_score}

"""
        file.write(result)

# Get test result of the best detect model



# Get validation results of classify runs
for run_path in classify_runs:
    config_path = os.path.join(run_path, 'args.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    imgsz = config['imgsz']
    model_path = os.path.join(run_path, 'weights/best.torchscript')

    # saves results under runs/classify/val*
    command = f"yolo val task=classify model={model_path} imgsz={imgsz} data=dataset/images"
    os.system(command=command)
