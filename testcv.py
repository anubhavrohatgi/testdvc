# import the opencv library 
import cv2 
import tensorflow as tf
import numpy as np

BLUR_THREHSOLD = 0.7
cv2.setNumThreads(1)
def detect_blur(image, sv_num=10):

    try:
       
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        s, u, v = tf.linalg.svd(img)
        #u, s, v = np.linalg.svd(img)
        top_sv = np.sum(s[0:sv_num])
        total_sv = np.sum(s)
        predicted_thresh = top_sv/total_sv
        # print(predicted_thresh)
        if predicted_thresh > BLUR_THREHSOLD:
            '''frame is blurred'''
            return True 
        return False
    except Exception as e:
        print(e)
        return False

# define a video capture object 
vid = cv2.VideoCapture(0) 

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 

	# Display the resulting frame 
	cv2.imshow('frame', frame) 
	
	print("Val : ",detect_blur(frame))
	
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

