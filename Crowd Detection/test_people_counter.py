from people_counter_core import PeopleCounter

counter = PeopleCounter(
    prototxt="path/to/deploy.prototxt",         # Path to your prototxt file
    model="path/to/model.caffemodel",           # Path to your model file
    confidence=0.4,                             # Detection confidence threshold
    skip_frames=30,                             # Set to 30 for most cases
    video_source="test_1.mp4"               # Path to your test video, or use 0 for webcam
)
counter.run(display=True)  # Set display=False if you don't want to show the output window