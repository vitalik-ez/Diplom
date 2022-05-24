from enum import Enum
import time


class PlayMode(Enum):
    STOP = 0
    PLAY = 1
    PAUSE = 2


class Controller:

    def __init__(self, view, model) -> None:
        self._view = view
        self._model = model

        #self.fps = 1000
        self.play_mode = PlayMode.STOP
        self._connectSignals()
        # self._video_stream()

    def _connectSignals(self):
        """Connect signals and slots."""
        self._view.ui.startButton.clicked.connect(lambda: self._start_stream())
        self._view.ui.pauseButton.clicked.connect(lambda: self._pause())
        self._view.ui.quitButton.clicked.connect(self._view.close)

    def _start_stream(self):
        self._view.frame_timer.timeout.connect(self._video_stream)
        self._view.frame_timer.start(int(1000 // 30))
        #self._model.video_stream.camera_stream()

    def _video_stream(self):
        processed_frame = self._model.stream_processing()
        self._view.display_video_stream(processed_frame)

    def _pause(self):
        self.play_mode = PlayMode.PAUSE
