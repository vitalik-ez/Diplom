import torch
import cv2


class TrafficSign:

    def __init__(self) -> None:
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path='weights/YoloV5s.pt')
        self.model.cuda()

    def detect(self, frame):
        self.model.conf = 0.4  # confidence threshold (0-1)
        self.model.iou = 0.2  # NMS IoU threshold (0-1)
        # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

        results = self.model(frame)  # custom inference size
        results = results.pandas().xyxy[0].to_dict('records')
        print(results)
        for box in results:
            #print(box)
            x1 = int(box['xmin'])
            y1 = int(box['ymin'])
            x2 = int(box['xmax'])
            y2 = int(box['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, box['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        return frame, results


if __name__ == '__main__':
    frame = cv2.imread('speed-limit-signs-stock-illustration_csp16016780.webp')
    tf = TrafficSign()
    cv2.imwrite("results.png", tf.detect(frame))
