import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
# starting_time = time.time()
# frame_id = 0

while True:
    _, frame = cap.read()
    # frame_id += 1

    height, width, channels = frame.shape
   


    #  # Tentukan posisi garis pembagi y
    line_y1 = height // 2
    
    # line_y2 = 2 * (height // 3)

    # # Gambar garis pembagi pada frame
    cv2.line(frame, (0, line_y1), (width, line_y1), (0, 255, 0), thickness=2)
    # cv2.line(frame, (0, line_y2), (width, line_y2), (0, 255, 0), thickness=2)
    # Tentukan posisi garis pembagi x
    line_x1 = width // 2
    # line_x2 = 2 * (width // 3)

    # Gambar garis pembagi vertikal pada frame
    cv2.line(frame, (line_x1, 0), (line_x1, height), (0, 255, 0), thickness=2)
    # cv2.line(frame, (line_x2, 0), (line_x2, height), (0, 255, 0), thickness=2)

    cv2.putText(frame, f'Ukuran per frame: {width} x {height}', (10, line_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # cv2.putText(frame, f'Total lebar frame: {width * 3} px', (10, line_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)



    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #0.2
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                print(center_x)
                # print(center_y)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if center_x > 320:
                    print("objek dikiri")
                elif center_x < 320:
                    print("objek dikanan")
                else :
                    print("objek gada")
            #    # Check if object detected is in any of the thirds of the frame
            #     if x > line_x1 and x < line_x2:
            #         text_x = 10
            #     elif x > line_x2:
            #         text_x = line_x2 + 10
            #     else:
            #         text_x = line_x1 + 10


            #     # Check which horizontal line the object is above
            #     if center_y < line_y1:
            #         text_y = line_y1 - 10
            #     elif center_y < line_y2:
            #         text_y = line_y2 - 10
            #     else:
            #         text_y = line_y2 + 10

                # if x > line_x1 :
                #     text_x = 30 
                # else: 
                #     text_x = line_x1 + 10

               

                # cv2.putText(frame, 'Ada objek', (text_x, line_y1 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #0.8 0.3
    
   
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)



    # elapsed_time = time.time() - starting_time
    # fps = frame_id / elapsed_time
    # cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)

    cv2.imshow("Image", frame)
    # frame = cv2.imread()
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()