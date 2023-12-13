# Importing necessary libraries
import cv2

# Setting up neural network
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


# Load all ids from coco file
ids = []
with open("coco.names", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        ids.append(class_name)

# Initalize camera and resolution
frame = cv2.VideoCapture(1)
frame.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
frame.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    # Retrives frame from webcam
    ret, img = frame.read()

    # Object detection
    id, score, boundries = model.detect(img)
    for id, score, boundries in zip(id, score, boundries):
        x, y, w, h = boundries
        class_name = ids[id]
        cv2.putText(img, (class_name), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.rectangle(img, (x, y), ((x+w), (y+h)), (0, 255, 0), 3)

    # Closes program
    cv2.imshow("Motion", img)
    key = cv2.waitKey(1)
    if key == ord('x'):
        break
