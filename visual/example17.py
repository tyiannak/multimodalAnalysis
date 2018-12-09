import cv2
import numpy

(Width, Height) = (640, 480)
cap = cv2.VideoCapture(0)
cap.set(3, Width); cap.set(4, Height)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Dif = 0
    if count > 0:
        Dif = numpy.abs(gray.astype(float) -
                        gray_prev.astype(float)).sum() / (128 * Height)
    count += 1
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:     # Show detected bounding boxes:
        x = int(x - w/6)
        y = int(y - h/6)
        w += int(w/6)
        h += int(h/6)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    if count > 1:
        for d in range(int(Dif)):
            cv2.putText(frame, "|" % Dif, (10+d, 10),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255))
    cv2.imshow('win', frame)
    gray_prev = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()