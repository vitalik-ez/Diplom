import torch
import cv2


class TrafficSign:

    def __init__(self) -> None:
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path='weights/YoloV5X_kaggle_dataset.pt')
        self.model.cuda()

    def detect(self, frame):
        self.model.conf = 0.25  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

        results = self.model(frame)  # custom inference size
        results = results.pandas().xyxy[0].to_dict('records')

        for box in results:
            #print(box)
            x1 = int(box['xmin'])
            y1 = int(box['ymin'])
            x2 = int(box['xmax'])
            y2 = int(box['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, box['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return frame


if __name__ == '__main__':
    frame = cv2.imread('test_data/speed_limit100.jpeg')
    tf = TrafficSign()
    cv2.imwrite("results.png", tf.detect(frame))
