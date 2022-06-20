from video_stream import VideoStream
from lane_detection import LaneDetection
from traffic_sign_detection import TrafficSign
import lane_detection_2
import cv2

class Model():

    def __init__(self) -> None:
        print("Model init...")
        self.video_stream = VideoStream()
        self.lane_detection = LaneDetection()
        self.traffic_sign = TrafficSign()

    def stream_processing(self):
        # VideoStream 
        frame = self.video_stream.get_frame()
        # Traffic sign detection
        result = self.traffic_sign.detect(frame)
        # Line detection
        try:
            #result = self.lane_detection.detect(frame)
            result = lane_detection_2.detect(frame)
        except Exception as ex:
            print("Road line not detected")
            result = frame
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