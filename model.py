from video_stream import VideoStream
from lane_detection import LaneDetection
from traffic_sign_detection import TrafficSign
import lane_detection_2


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
        return result
