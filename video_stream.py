import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')

from gi.repository import GstSdp
from gi.repository import GstWebRTC
from gi.repository import Gst, GObject, GLib
import sys
import cv2
import numpy as np
import os
import time



Gst.init(None)

#video_path = "test_data/dubai_traffic_sign.mp4"
video_path = 'test_data/test7.mp4'

'''
def on_new_sample(app_sink):
    print("on_new_sample")
    sample = app_sink.pull_sample()
    caps = sample.get_caps()

    # Extract the width and height info from the sample's caps
    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")

    # Get the actual data
    buffer = sample.get_buffer()
    print(caps,"buffer size ",buffer.get_size())
    # Get read access to the buffer data
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if not success:
        raise RuntimeError("Could not map buffer data!")

    numpy_frame = np.ndarray(
        shape=(height, width, 3),
        dtype=np.uint8,
        buffer=map_info.data)

    buffer.unmap(map_info)
'''


class VideoStream:

    def __init__(self):
        #self.mainloop = GLib.MainLoop()
        self.pipeline = Gst.Pipeline.new('stream')
        self._frame = self.opencv_stream()
        #self._frame = np.zeros((720, 1280, 3), dtype = np.uint8)

    def get_frame(self):
        return next(self._frame)
        # return self._frame
        # return np.random.randint(255, size=(480,640,3), dtype=np.uint8)

    def message(self, bus, message):
        mtype = message.type
        if mtype == Gst.MessageType.EOS:
            print("EOS")
            self.exit()
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print('Error: %s' % err, debug)
        # self.exit()

    def gst_to_opencv(self, sample):
        buf = sample.get_buffer()
        buff = buf.extract_dup(0, buf.get_size())

        caps = sample.get_caps()
        height = caps.get_structure(0).get_value('height')
        width = caps.get_structure(0).get_value('width')

        print("height", height, "width", width, buf.get_size())

        arr = np.ndarray((height, width, 3),
                         buffer=buff,
                         dtype=np.uint8)
        return arr

    def new_buffer(self, sink, data):
        sample = sink.emit("pull-sample")
        self._frame = self.gst_to_opencv(sample)
        return Gst.FlowReturn.OK

    def camera_stream(self):
        #self.camera = Gst.ElementFactory.make('v4l2src', 'v4l2src')

        self.mainloop = GLib.MainLoop()
        self.camera = Gst.ElementFactory.make('filesrc', 'filesrc')
        self.camera.set_property('location', video_path)

        self.decode = Gst.ElementFactory.make('decodebin', 'decode')

        self.videorate = Gst.ElementFactory.make('videorate', 'videorate')
        self.capsfilter = Gst.ElementFactory.make('capsfilter', 'capsfilter')

        caps_speed = Gst.caps_from_string("video/x-raw,framerate=15/1")
        self.capsfilter.set_property('caps', caps_speed)

        self.queue = Gst.ElementFactory.make('queue', 'queue')
        self.videoconvert = Gst.ElementFactory.make(
            'videoconvert', 'videoconvert')

        self.sink = Gst.ElementFactory.make('appsink', 'sink')
        self.sink.set_property("emit-signals", True)
        self.sink.set_property("enable-last-sample", False)
        self.sink.set_property("sync", False)
        self.sink.set_property("drop", True)
        self.sink.set_property("async", True)
        self.sink.set_property("max-buffers", 2)

        caps = Gst.caps_from_string(
            "video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg},framerate=5/1")
        self.sink.set_property("caps", caps)

        self.sink.connect("new-sample", self.new_buffer, self.sink)

        self.pipeline.add(self.camera)
        self.pipeline.add(self.decode)
        self.pipeline.add(self.videorate)
        self.pipeline.add(self.capsfilter)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.sink)

        self.camera.link(self.decode)

        def decodebin_src_pad_created(element, pad):
            self.decode.link(self.videorate)

        self.decode.connect("pad-added", decodebin_src_pad_created)
        self.videorate.link(self.capsfilter)
        self.capsfilter.link(self.queue)
        self.queue.link(self.videoconvert)
        self.videoconvert.link(self.sink)

        self.pipeline.set_state(Gst.State.PLAYING)

        self.mainloop.run()

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message', self.message)

    def opencv_stream(self):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video file")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            yield frame


if __name__ == "__main__":
    video = VideoStream()
    #video.camera_stream()

    '''
    gen = video.opencv_stream()
    frame = next(gen)

    print(frame.shape)
    '''
    