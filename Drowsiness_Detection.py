from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
import time

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
	A = distance.euclidean(eye[1], eye[5]) # distance between  the top and bottom of the eye
	B = distance.euclidean(eye[2], eye[4]) # distance between  the top and bottom of the eye
	C = distance.euclidean(eye[0], eye[3]) # distance between left and right of the eye
	ear = (A + B) / (2.0 * C)
	return ear

#intialize the firebase app
cred = credentials.Certificate("./timeturner-key.json")
firebase_admin.initialize_app(cred)

#initialize the firestore database
db = firestore.client()

#initialize user's document in the database
userId = input("Enter your email (the one that you use with timeturner):   ")
doc_ref = db.collection(u'users').document(userId)

#thresholds to check if the user is drowsy, this will only be used for demo purposes
#actual threshold will be calculated on front end of website
low_thresh = 0.28
high_thresh = 0.35

#use dlib's face detector and then create the facial landmark predictor with model from https://github.com/akshaybahadur21/Drowsiness_Detection
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code 

#left eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
#right eye landmarks
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#initialize opencv's video capture
cap=cv2.VideoCapture(0)
#add a counter to keep track of frames
flag=0
while True:
	frames = 0
	earSum = 0
	time_interval = time.time()+ 10 #time interval in seconds
	while time.time() < time_interval:
		
		#read the frame
		ret, frame=cap.read()
		frame = imutils.resize(frame, width=450)
		#convert to grayscale is better for efficiency
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = detect(gray, 0)
		#loop over each face detected
		for subject in subjects:
			#get the landmarks/parts for the face in the subject
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)#converting to NumPy Array
			#extract the left and right eye coordinates
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			#compute the eye aspect ratio for both eyes
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			#average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			#draw the contours on the frame
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			#check if the EAR is less than the threshold
			if ear < low_thresh:
				flag += 1
				cv2.putText(frame, "EAR: "+ str(ear) +" - Low Energy", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				cv2.putText(frame, "Low Energy", (10, 230),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				frames+=1
				earSum+=ear		
			elif ear >= low_thresh and ear<high_thresh:
				cv2.putText(frame, "EAR: "+ str(ear), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				cv2.putText(frame, "Medium Energy", (10, 230),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				frames+=1
				earSum+=ear
					
			else:
				cv2.putText(frame, "EAR: "+ str(ear) , (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "High Energy", (10, 230),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				frames+=1
				earSum+=ear
			

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	#push ear to firebase
	doc_ref.set({
		'EAR': earSum/frames
	}, merge=True)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 
