from arlo import Arlo

from subprocess import call
import cv2
USERNAME = 'hoyt@layson.net'
PASSWORD = '$$$dRewjEff1996'


def process_image(frame):
	process_frame = frame.copy()
	print(process_frame.shape)
	process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
	process_frame = cv2.resize(process_frame,(int(300),int(100)))
	return process_frame





try:

    # Instantiating the Arlo object automatically calls Login(),
    # which returns an oAuth token that gets cached.
    # Subsequent successful calls to login will update the oAuth token.
    arlo = Arlo(USERNAME, PASSWORD)
    # At this point you're logged into Arlo.

    # Get the list of devices and filter on device type to only get the cameras.
    # This will return an array which includes all of the canera's associated metadata.
    cameras = arlo.GetDevices('camera')


    # Get the list of devices and filter on device type to only get the basestation.
    # This will return an array which includes all of the basestation's associated metadata.
    basestations = arlo.GetDevices('basestation')
    
    # Send the command to start the stream and return the stream url.
    url = arlo.StartStream(basestations[0], cameras[1]) #either 1 or 4 working
    vs = cv2.VideoCapture(url)
    while True:
        _, frame = vs.read()
        process_frame = process_image(frame)
        cv2.imshow('test', process_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


except Exception as e:
    print(e)