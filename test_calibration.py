import calibration
import cv2
import glob
import pickle

if __name__=='__main__':
    # Load pickled camera calibration
    data = pickle.load(open("calibration.p", 'rb'))
    mtx = data["mtx"]
    dist = data["dist"]

    for filename in glob.glob("./test_images/*.jpg"):
        test_img = cv2.imread(filename)
        undist = cv2.undistort(test_img, mtx, dist, None, mtx)
        cv2.imshow('Before', test_img)
        cv2.imshow('After', undist)
        cv2.waitKey(0)