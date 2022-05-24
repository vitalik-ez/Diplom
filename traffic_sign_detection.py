import torch
import cv2


class TrafficSign:

    def __init__(self) -> None:
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path='best.pt')

    def detect(self, frame):
        self.model.conf = 0.25  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        '''
        self.model.classes = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
                                'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
                                'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 
                                'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection', 
                                'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 
                                'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 
                                'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 
                                'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 
                                'Wild animals crossing', 'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 
                                'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 
                                'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons']
        '''
        

        results = self.model(frame)  # custom inference size
        print(results.pandas().xyxy[0])
        for box in results.xyxy[0]:
            print(box)
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imshow("OutputWindow", frame)
        cv2.waitKey(0)


if __name__ == '__main__':
    frame = cv2.imread('test_data/test.jpg')
    tf = TrafficSign()
    tf.detect(frame)
