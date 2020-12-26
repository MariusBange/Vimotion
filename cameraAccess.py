'''camera object and threading for camera capturing'''
import cv2
import threading
import os
from schedule import set_delete_timer

class RecordingThread (threading.Thread):
    '''handle camera capturing'''
    def __init__(self, camera):
        '''sets resolution of camera and name of recording'''
        threading.Thread.__init__(self)
        self.isRunning = True

        self.cap = camera
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH);
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        filename = "webcamRecording(1)"
        i = 1

        while os.path.isfile('static/uploads/video/' + filename + ".mp4"):
            ending = len(str(i)) + 2
            i += 1
            filename = filename[:-ending] + '(' + str(i) + ')'

        self.filename = filename + '.mp4'
        self.out = cv2.VideoWriter('./static/uploads/video/' + filename + '.mp4',fourcc, 15.0, (int(w),int(h)))

    def run(self):
        '''saves/writes frame while recording'''
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)#cv2.flip(frame, 1) for mirrored video

        self.out.release()

    def stop(self):
        '''sets recording status to false'''
        self.isRunning = False

    def __del__(self):
        '''destroy output stream'''
        self.out.release()

class VideoCamera(object):
    '''Camera object'''
    def __init__(self):
        '''initialization'''
        # Open a camera
        self.cap = cv2.VideoCapture(0)

        # Initialize video recording environment
        self.is_record = False
        self.out = None
        self.name = None

        # Thread for recording
        self.recordingThread = None

    def __del__(self):
        '''destroy video capturing'''
        self.cap.release()

    def get_frame(self):
        '''get frame from capture'''
        ret, frame = self.cap.read()

        if ret:
            image = cv2.flip(frame, 1)
            ret, jpeg = cv2.imencode('.jpg', image)

            return jpeg.tobytes()

        else:
            return None

    def start_record(self):
        '''set name of recording as camera name and start recording'''
        self.is_record = True
        self.recordingThread = RecordingThread(self.cap)
        self.name = self.recordingThread.filename
        self.recordingThread.start()
        set_delete_timer('static/uploads/video/' + self.name)

    def stop_record(self):
        '''stops recording'''
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
