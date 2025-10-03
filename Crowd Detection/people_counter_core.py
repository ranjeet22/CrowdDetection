import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import dlib
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject

class PeopleCounter:
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    def __init__(self, prototxt, model, confidence=0.4, skip_frames=30, video_source=0):
        self.prototxt = prototxt
        self.model = model
        self.confidence = confidence
        self.skip_frames = skip_frames
        self.video_source = video_source  # 0 for webcam, or path to video file

        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        self.W = None
        self.H = None

        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0

    def run(self, display=True):
        # Video source: file or webcam
        if isinstance(self.video_source, str):
            vs = cv2.VideoCapture(self.video_source)
            get_frame = lambda: vs.read()[1]
        else:
            vs = VideoStream(self.video_source).start()
            get_frame = lambda: vs.read()

        while True:
            frame = get_frame()
            if frame is None:
                break

            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]

            status = "Waiting"
            rects = []

            if self.totalFrames % self.skip_frames == 0:
                status = "Detecting"
                self.trackers = []

                blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.confidence:
                        idx = int(detections[0, 0, i, 1])
                        if self.CLASSES[idx] != "person":
                            continue
                        box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                        (startX, startY, endX, endY) = box.astype("int")

                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)
                        self.trackers.append(tracker)
            else:
                for tracker in self.trackers:
                    status = "Tracking"
                    tracker.update(rgb)
                    pos = tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    rects.append((startX, startY, endX, endY))

            # Draw center line
            cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 0, 0), 2)
            # update centroid tracker
            objects = self.ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = self.trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    if not to.counted:
                        if direction < 0 and centroid[1] < self.H // 2:
                            self.totalUp += 1
                            to.counted = True
                        elif direction > 0 and centroid[1] > self.H // 2:
                            self.totalDown += 1
                            to.counted = True
                self.trackableObjects[objectID] = to
                text = f"ID {objectID}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            # Display info
            info = [
                ("Exit", self.totalUp),
                ("Enter", self.totalDown),
                ("Status", status),
                ("Total inside", self.totalDown - self.totalUp)
            ]
            for (i, (k, v)) in enumerate(info):
                cv2.putText(frame, f"{k}: {v}", (10, self.H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if display:
                cv2.imshow("People Counter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            self.totalFrames += 1

        if isinstance(self.video_source, str):
            vs.release()
        else:
            vs.stop()
        cv2.destroyAllWindows()