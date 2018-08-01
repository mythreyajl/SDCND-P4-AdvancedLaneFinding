import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def extract_corners(img, nx, ny):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners from image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    return ret, corners


class Calibration:
    def __init__(self, path="./camera_cal/*.jpg", nx=9, ny=6):
        self.path = path
        self.obj_points = []
        self.img_points = []
        self.nx = nx
        self.ny = ny
        self.objp = np.zeros((nx * ny, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        self.img_shape = None

    def calibrate_camera(self):
        for filename in glob.glob(self.path):
            # Open file
            img = cv2.imread(filename)

            # Set shape
            if not self.img_shape:
                self.img_shape = img.shape[1::-1]

            # Extract corners
            ret, corners = extract_corners(img, self.nx, self.ny)

            if ret:  # append corners to list of image corners
                #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                self.img_points.append(corners)
                self.obj_points.append(self.objp)
                #cv2.imshow('Corners', img)
                #cv2.waitKey(0)

        # Return calibrated camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.img_shape, None, None)
        if ret:
            print("---Calibrating camera complete.")
        else:
            print("---Error. Calibration failed")
        return ret, mtx, dist, rvecs, tvecs

    def store_calib(self):
        ret, mtx, dist, rvecs, tvecs = self.calibrate_camera()
        if ret:
            data = {"mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
            with open('calibration.p', 'wb') as f:
                pickle.dump(data, f)
            print("---Pickling data complete.")


if __name__=="__main__":
    path = "./camera_cal/*.jpg"

    # prepare object points
    nx = 9
    ny = 6

    calib = Calibration(path, nx, ny)
    calib.store_calib()
    #ret, mtx, dist, rvecs, tvecs = calib.calibrate_camera()
    #for filename in glob.glob(path):
    #    test_img = cv2.imread(filename)
    #    undist = cv2.undistort(test_img, mtx, dist, None, mtx)
    #    cv2.imshow("Distorted image pre-calibration", test_img)
    #    cv2.imshow("Undistorted image post-calibration", undist)
    #    cv2.waitKey(0)







