import face_recognition as fr
import cv2


known_face_encoding = []
known_face_name = []

# opencv module reading image.
person1_image = cv2.imread("salman.jpg")
person2_image =cv2.imread("shahrukh.jpg")

# converting BGR image to RGB.
rgb_image1 = cv2.cvtColor(person1_image, cv2.COLOR_BGR2RGB)
rgb_image2 = cv2.cvtColor(person2_image, cv2.COLOR_BGR2RGB)

#encoding the face in image.
person1_encoding = fr.face_encodings(rgb_image1)[0]
person2_encoding = fr.face_encodings(rgb_image2)[0]

#appending encodings and face name to the list.
known_face_encoding.append(person1_encoding)
known_face_encoding.append(person2_encoding)

known_face_name.append("salman")
known_face_name.append("shahrukh")

#initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    #capture frame by frame.
    ret, frame = video_capture.read()
    
    #find all frame location in current frame and encodes it.
    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)
    

    #loop through each face found in the frame.
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        #check if the face matches with any known face
        matches = fr.compare_faces(known_face_encoding,  face_encoding)
        name = "unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_name[first_match_index]

        #draw the box around the face and label with the names.
        cv2.rectangle(frame, (left, top), (right, bottom),(0 ,0, 255), 2)
        cv2.putText(frame, name, (left, top -10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),2)

    #display the frame
    cv2.imshow("Video", frame)

    #break the loop when the "q" key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the webcam and close opencv windows.
video_capture.release()
cv2.destroyAllWindows()
