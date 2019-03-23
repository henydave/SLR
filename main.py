import cv2
import timeit
import numpy as np
from utils.image_transformation import transform 
from utils.image_transformation import write_text
from utils.load_data import process_image
import pickle
from sklearn.metrics import adjusted_rand_score
from utils.predict_labels import predict_label_correlation,predict_label_svm
def Image(path):
    frame = cv2.imread(path)
    o_frame = transform(frame)
    label = predict_label_svm(frame)
    
    annotated_img = write_text(frame,label)
    print(label)
    cv2.imshow('input',annotated_img)
    cv2.imshow('output',o_frame)
    cv2.waitKey(0)

def Video():
    cap = cv2.VideoCapture(0)
    while not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cv2.waitKey(1000)
    label = ''
    cntr = 1
    while True:
        flag, frame = cap.read()
        if flag:
            frame = cv2.flip(frame,1)
            cv2.rectangle(frame,(400,100),(600,300),(255,0,0),2)
            annotated_frame = write_text(frame, label)
            cv2.imshow('input',annotated_frame)
                #cv2.imshow('input',frame[100:300,400:600])
            
        else:
            cv2.waitKey(1000)
        k = cv2.waitKey(5)
        if k == 27:
            break
        elif k == ord('c'):
            start = timeit.default_timer()
            #o_frame = trancsform(frame[100:300,400:600])
            label = predict_label_svm(frame[100:300,400:600])
            print(label)
            stop = timeit.default_timer()
            rt = stop-start
            print('Time: ',rt)
            #cv2.imshow('output',o_frame)
        elif k == ord('s'):
            cv2.imwrite('images/'+str(cntr)+'.jpg',frame[100:300,400:600])
            cntr += 1
            
def test_transformation():
    Image('G:/Sgp2/SLR/SLR/images/TEST/004.jpg')


if __name__ == "__main__":
    #test_transformation()
    Video()
    cv2.destroyAllWindows()
