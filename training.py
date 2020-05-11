import cv2      #importing OpenCV
import numpy as np #importing numpy as np

cap = cv2.VideoCapture(0) #Capturing Video from webcam # 0 represent the default camera of system
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") #using haarcascade xml file

skip = 0
face_data = []
dataset_path = './data/'
file_name=input("Enter the name of person :")

while True :
    ret,frame = cap.read() # string the caputring video frame

    if ret == False : #if no frame captured then loop is continued
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # converting to grayscale to save storage

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3]) 
    for face in faces[-1:] :
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) 
        
       #Extraction of required face area
        offset = 10
        face_section = frame[y-offset:y+offset+h , x-offset:x+offset+w]
        face_section = cv2.resize(face_section,(100,100))
        
        skip = skip + 1 #Storing every 10th frame and skipping the rest
        if skip%10 == 0 :
            face_data.append(face_section)
            print(len(face_data))
    
    cv2.imshow("Frame with rectangle",frame)
    cv2.imshow(" Face Section",face_section)    

    key_pressed = cv2.waitKey(1) & 0xFF 
    if key_pressed == ord('p') : #terminate the program when q is pressed
        break  
        
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data)  
print("Data successfully saved at"+dataset_path+file_name+'.npy')              

cap.release()         
cv2.destroyAllWindows()            
