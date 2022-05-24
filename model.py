from video_stream import VideoStream
from lane_detection import LaneDetection


class Model():

    def __init__(self) -> None:
        print("Model init...")
        self.video_stream = VideoStream()
        self.lane_detection = LaneDetection()

    def stream_processing(self):
        # VideoStream 
        result = self.video_stream.get_frame()
        # Line detection
        #result = self.lane_detection.detect(frame)
        
        # Traffic sign detection
        return result
