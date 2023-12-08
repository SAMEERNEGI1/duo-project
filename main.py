import cv2

car=cv2.VideoCapture(r"C:\Users\samee\OneDrive\Desktop\vehicle detect\car video.mp4")
car_haar=cv2.CascadeClassifier(r"C:\Users\samee\OneDrive\Desktop\vehicle detect\car algo.xml")
while True:

    ret,frames=car.read()
    gray=cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    cars=car_haar.detectMultiScale(gray,1.1,1)
    for(x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,0),1)

    cv2.imshow('video', frames)
    if cv2.waitKey(33)==27:
        break


cv2.destroyAllWindows()