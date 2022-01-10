import torchvision
import cv2
import torch
import numpy as np
def plot_box(img, box, box_type="xywh", color=(255, 0, 0)):
    """在图片上绘制矩形框"""

    if box_type == "xywh":
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
    elif box_type == "xyxy":
        x1, y1, x2, y2 = box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
def imshow(title, img, wait_ms=10):
    """显示图像 Esc键退出"""

    if wait_ms <= 0:
        while True:
            cv2.imshow(title, img)
            if cv2.waitKey(20) == 27:  # Esc
                break
    else:
        cv2.imshow(title, img)
        cv2.waitKey(wait_ms)
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [
        COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]["labels"].cpu().numpy())
    ]  # Get the Prediction Score
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].detach().cpu().numpy())
    ]  # Bounding boxes
    pred_score = list(pred[0]["scores"].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_classes = pred_classes[: pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == "person":
            person_boxes.append(box)

    return person_boxes

if __name__ == "__main__":

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.cuda()
    box_model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, image_bgr = cap.read()

        input = []
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().cuda()
        input.append(img_tensor)

        box_list = get_person_detection_boxes(box_model, input, threshold=0.9)

        for box in box_list:
            (x1, y1), (x2, y2) = box
            box = [x1, y1, x2, y2]
            # print(box)
            plot_box(image_bgr, box, "xyxy")
        print("box", box, type(box))
        imshow("img", image_bgr)