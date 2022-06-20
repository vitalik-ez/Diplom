from cProfile import label
from video_stream import VideoStream
from lane_detection import LaneDetection
from traffic_sign_detection import TrafficSign
import lane_detection_2
import cv2
import numpy as np

class Model():

    def __init__(self) -> None:
        print("Model init...")
        self.video_stream = VideoStream()
        self.lane_detection = LaneDetection()
        self.traffic_sign = TrafficSign()
        self._detected_traffic_sign = []
        self._traffic_sign_image = {}

    def stream_processing(self):
        # VideoStream 
        frame = self.video_stream.get_frame()
        # Traffic sign detection
        result, traffic_signs = self.traffic_sign.detect(frame)
        # Line detection
        

        try:
            #result = self.lane_detection.detect(frame)
            result = lane_detection_2.detect(frame)
        except Exception as ex:
            print("Road line not detected")
            result = frame
        
        
        result = self.show_traffics_sign(traffic_signs, result)
        return result


    def show_traffics_sign(self, traffic_signs, frame):
        if self._detected_traffic_sign:
            for sign in traffic_signs:
                presense_traffic_sign = list(filter(lambda detected_traffic_sign: detected_traffic_sign['name'] == sign['name'], self._detected_traffic_sign))
                if not presense_traffic_sign:
                    self._detected_traffic_sign += sign
        else:
            self._detected_traffic_sign += traffic_signs

        
        for sign in self._detected_traffic_sign:
            if sign['name'] not in self._traffic_sign_image:
                self._traffic_sign_image[sign['name']] = cv2.cvtColor(frame[int(sign['ymin']):int(sign['ymax']), int(sign['xmin']):int(sign['xmax'])], cv2.COLOR_BGR2RGB)

        traffic_sign_size = (100,100)
        images = list(self._traffic_sign_image.values())
        labels = list(self._traffic_sign_image.keys())
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], traffic_sign_size, interpolation = cv2.INTER_AREA)

            text = np.zeros([50,100,3],dtype=np.uint8)
            text.fill(255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontColor = (255,0,0)
            thickness = 1
            lineType = 2
            cv2.putText(text, labels[i], (10,20),  font, fontScale, fontColor, thickness, lineType)
            cv2.imwrite('text.png', text)
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            images[i] = cv2.vconcat([images[i], text])

        traffic_sign_column = cv2.vconcat(images)
        #cv2.imwrite('sign.png', traffic_sign_column)


        h1, w1 = frame.shape[:2]
        h2, w2 = traffic_sign_column.shape[:2]

        result = np.zeros((max(h1, h2), w1+w2,3), dtype=np.uint8)
        result[:,:] = (255,255,255)

        result[:h1, :w1,:3] = frame
        result[:h2, w1:w1+w2,:3] = traffic_sign_column

        #frame = cv2.hconcat([frame, traffic_sign_column])
        cv2.imwrite('sign.png', result)
        return result



if __name__ == '__main__':
    model = Model()

    import cv2
    import time
    video_path = 'test_data/test7.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file")
        exit()


    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    fps = 20
    output = cv2.VideoWriter('result_road_line_detection.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)

    count = 0
    detect_line_time = 0
    detect_traffic_sign_time = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        count += 1

        start_time = time.time()
        frame = model.traffic_sign.detect(frame)
        detect_traffic_sign_time += time.time() - start_time

        try:
            #result = self.lane_detection.detect(frame)
            start_time = time.time()
            result = lane_detection_2.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detect_line_time += time.time() - start_time
        except Exception as ex:
            print("Road line not detected")
            result = frame
        output.write(result)
        print(f"Frame {count} processed")

    cap.release()
    output.release()
            
    print("=============TEST=============")
    print(f"Detect road line. Processing one frame takes {round(detect_line_time/count, 4)} seconds")
    print(f"Detect traffic sign. Processing one frame takes {round(detect_traffic_sign_time/count, 4)} seconds")