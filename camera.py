import cv2
import numpy as np
from keras.models import load_model

camara = cv2.VideoCapture(0)

WIDTH=52
HEIGHT=52
CAMERA_HEIGHT = 500
NAME_CLASSES = ['Speed limit (30km/h)', 'Speed limit (80km/h)', 'Stop', 'Vehicles heavy prohibited', 'Slippery road', 'Pedestrians', 'Bicycles crossing']

model = load_model('saved_models/keras_traffic_signs_trained_model.h5')

while(True):
    # read a new frame
    _, frame = camara.read()

    # flip the frame
    frame = cv2.flip(frame,1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * CAMERA_HEIGHT) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, CAMERA_HEIGHT))

    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)

    # get ROI
    roi = frame[75+2:425-2, 300+2:650-2]

    # parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # resize
    roi = cv2.resize(roi, (WIDTH, HEIGHT))

    # predict!    
    roi = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi)
    type_1, type_2, type_3, type_4, type_5, type_6, type_7= predictions[0]

    # add text
    type_1_text = '{}: {}%'.format(NAME_CLASSES[0], int(type_1*100))
    cv2.putText(frame, type_1_text, (10, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # add text
    tipe_2_text = '{}: {}%'.format(NAME_CLASSES[1], int(type_2*100))
    cv2.putText(frame, tipe_2_text, (10, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # add text
    tipe_3_text = '{}: {}%'.format(NAME_CLASSES[2], int(type_3*100))
    cv2.putText(frame, tipe_3_text, (10, 230), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # add text
    tipe_4_text = '{}: {}%'.format(NAME_CLASSES[3], int(type_4*100))
    cv2.putText(frame, tipe_4_text, (10, 260), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # add text
    tipe_5_text = '{}: {}%'.format(NAME_CLASSES[4], int(type_5*100))
    cv2.putText(frame, tipe_5_text, (10, 290), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # add text
    tipe_6_text = '{}: {}%'.format(NAME_CLASSES[5], int(type_6*100))
    cv2.putText(frame, tipe_6_text, (10, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # add text
    tipe_7_text = '{}: {}%'.format(NAME_CLASSES[6], int(type_7*100))
    cv2.putText(frame, tipe_7_text, (10, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # show the frame
    cv2.imshow("Capturing frames", frame)

    key = cv2.waitKey(1)

    # quit camera if 'q' key is pressed
    if key & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()