from ui_main_window import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import qimage2ndarray
# MainWindow
class View(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Setup video stream container
        self.video_stream_container()
        #self.setup_camera()


    def video_stream_container(self, width = 1280, height = 720):
        self.video_size = QtCore.QSize(width, height)
        self.camera_capture = cv2.VideoCapture(0)
        
        # take frame from video capture
        self.frame_timer = QtCore.QTimer()

        # Label to show process frame (frame_label)
        self.ui.videoStream.setFixedSize(self.video_size)

    
    # Remove method
    def setup_camera(self):
        self.camera_capture.set(3, self.video_size.width())
        self.camera_capture.set(4, self.video_size.height())

        self.frame_timer.timeout.connect(self.display_video_stream)
        self.frame_timer.start(int(1000 // self.fps))


    def display_video_stream(self, frame):
        image = qimage2ndarray.array2qimage(frame)
        self.ui.videoStream.setPixmap(QtGui.QPixmap.fromImage(image))
        
