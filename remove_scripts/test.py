import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import os

Gst.init(None)
mainloop = GObject.MainLoop()

# Short way:
# pipeline = Gst.parse_launch("filesrc name=filesource ! decodebin ! autovideosink")
# pipeline.get_by_name("filesource").set_property("location", "background.mov")

# Long way:
pipeline = Gst.Pipeline.new("pipeline")

filesrc = Gst.ElementFactory.make("filesrc", "filesrc")
filesrc.set_property("location", "/home/vitalii/Desktop/DriverAssistance/test_data/german.mp4")
pipeline.add(filesrc)

decodebin = Gst.ElementFactory.make("decodebin", "decodebin")
pipeline.add(decodebin)

autovideosink = Gst.ElementFactory.make("autovideosink", "autovideosink")
pipeline.add(autovideosink)

filesrc.link(decodebin)

def decodebin_src_pad_created(element, pad):
    decodebin.link(autovideosink)

decodebin.connect("pad-added", decodebin_src_pad_created)

# Start playing.
pipeline.set_state(Gst.State.PLAYING)

def message(bus, message):
    if message.type == Gst.MessageType.EOS:
        pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect('message', message)

print("before mainloop run")
mainloop.run()
print("after mainloop run")