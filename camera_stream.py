import sys
import cv2
import numpy as np
import os
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp


Gst.init(None)


video_path = "/home/vitalii/Desktop/DriverAssistance/test_data/german.mp4"


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

class VideoStream:

    def __init__(self):
        self.mainloop = GLib.MainLoop()
        self.pipeline = Gst.Pipeline.new('stream')
        
        

    # Callback für decode.connected
    def decode_src_created(self, element, pad):
        pad.link(self.queuevideo.get_static_pad('sink'))
        pad.link(self.queueaudio.get_static_pad('sink'))

    def play(self):
        print("PLAY")
        self.playmode = "play"
        self.pipeline.set_state(Gst.State.PLAYING)

    def pause(self):
        self.playmode = "pause"
        self.pipeline.set_state(Gst.State.PAUSED)

    def setFile(self, filesrc):
        self.audioconvert.unlink(self.audiosink)
        self.videoconvert.unlink(self.videosink)
        self.pipeline.set_state(Gst.State.READY)
        if os.path.isfile(filesrc):
            print("Setting File to %s" % filesrc)
            self.filesrc.set_property('location', filesrc)
        else:
            print("Setting File to Pause")
            self.filesrc.set_property('location', video_path)
        self.audioconvert.link(self.audiosink)
        self.videoconvert.link(self.videosink)
        self.play()

    def stop(self):
        self.playmode = "stop"
        self.pipeline.set_state(Gst.State.NULL)


    def message(self, bus, message):
        mtype = message.type
        if mtype == Gst.MessageType.EOS:
            print("EOS")
            self.exit()
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print('Error: %s' % err, debug)
        #self.exit()
        print("Else message")



    def test_stream(self):
        self.filesrc = Gst.ElementFactory.make('filesrc', 'filesrc')
        self.filesrc.set_property('location', video_path)

        self.decode = Gst.ElementFactory.make('decodebin', 'decode')
        self.queue = Gst.ElementFactory.make('queue', 'queue')
        self.videoconvert = Gst.ElementFactory.make('videoconvert', 'videoconvert')

        self.sink = Gst.ElementFactory.make('autovideosink', 'sink')
        #self.sink.connect("new-sample", on_new_sample)

        self.pipeline.add(self.filesrc)
        self.pipeline.add(self.decode)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.sink)

        self.filesrc.link(self.decode)

        def decodebin_src_pad_created(element, pad):
            self.decode.link(self.queue)

        self.decode.connect("pad-added", decodebin_src_pad_created)
        self.queue.link(self.videoconvert)
        self.videoconvert.link(self.sink)

    
        self.pipeline.set_state(Gst.State.PLAYING)
        
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()   
        self.bus.connect('message', self.message) 

        self.mainloop.run()


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
        self.arr = self.gst_to_opencv(sample)
        cv2.imwrite("result.png", self.arr)
        return Gst.FlowReturn.OK

    def camera_stream(self):
        #self.camera = Gst.ElementFactory.make('v4l2src', 'v4l2src')

        self.camera = Gst.ElementFactory.make('filesrc', 'filesrc')
        self.camera.set_property('location', video_path)



        self.decode = Gst.ElementFactory.make('decodebin', 'decode')
        self.queue = Gst.ElementFactory.make('queue', 'queue')
        self.videoconvert = Gst.ElementFactory.make('videoconvert', 'videoconvert')

        self.sink = Gst.ElementFactory.make('appsink', 'sink')
        self.sink.set_property("emit-signals", True)
        self.sink.set_property("enable-last-sample", False)
        self.sink.set_property("sync", False)
        self.sink.set_property("drop", True)
        self.sink.set_property("async", True)
        self.sink.set_property("max-buffers", 2)

        caps = Gst.caps_from_string("video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg},framerate=30/1")
        self.sink.set_property("caps", caps)

        self.sink.connect("new-sample", self.new_buffer, self.sink)
        

        self.pipeline.add(self.camera)
        self.pipeline.add(self.decode)
        self.pipeline.add(self.queue)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.sink)

        self.camera.link(self.decode)

        def decodebin_src_pad_created(element, pad):
            self.decode.link(self.queue)

        self.decode.connect("pad-added", decodebin_src_pad_created)
        self.queue.link(self.videoconvert)
        self.videoconvert.link(self.sink)

    
        self.pipeline.set_state(Gst.State.PLAYING)
        
        
        self.mainloop.run()

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()   
        self.bus.connect('message', self.message) 


if __name__ == "__main__":
    video = VideoStream()

    #video.test_stream()
    video.camera_stream()


   