# USAGE
# python Blur_detector.py --images [image path]
# import the necessary packages
from imutils import paths
import argparse
import cv2
import numpy as np
import pywt
import csv
import os
import subprocess
from subprocess import Popen

def w2d(img, mode='haar', level=1):
    kernel_size = 3
    scale = 1
    delta = 0
    ddepth = cv2.CV_16UC3

    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    #print(imArray_H)
    blur = cv2.GaussianBlur(imArray_H,(3,3),0,0, cv2.BORDER_DEFAULT)
    laplacian = cv2.Laplacian(blur, ddepth = ddepth, ksize = kernel_size, scale = scale,delta = delta, borderType=cv2.BORDER_DEFAULT)
    (mean, stddev) = cv2.meanStdDev(laplacian)
    laplacian_operator_std_dev = stddev[0]
    laplacian_operator_variance = stddev[0]*stddev[0]
    return(laplacian_operator_variance)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=3500.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# loop over the input images
os.remove('filename.csv')
open_file = open("filename.csv", "a")
columnTitleRow = "BPLR_Name,BlurType,Laplacian Variance, ImageFileNumber\n"
open_file.write(columnTitleRow)
for imagePath in paths.list_images(args["images"]):

	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	#print(imagePath)
	image = cv2.imread(imagePath)
	laplac_variance = w2d(image,'db1',7)
	laplac_variance2 = int(laplac_variance)
	
	xl = str(imagePath)
	(xl1,xl2) = xl.split("/",1)
	(xl3,xl4) = xl2.split("image",1)
	(xl5,xl6) = xl4.split(".",1)
	BPLR = "R1"
	#imagenumber = 30
	if int(xl5) <=30:
		BPLR = "R3"
	elif int(xl5) >30 and int(xl5) <= 60:
		BPLR ="R1_BLUR"
	else:
		BPLR = "R2"
	
	
	text = "Not-Blurry"

	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if laplac_variance > args["threshold"]:
		text = "Blurry"
	
	filewriter = csv.writer(open_file)
	filewriter.writerow([BPLR, text, laplac_variance,xl5])
	#print(xl4 + " " + xm + "  " +  text)
	print(text + "  " + str(laplac_variance2)  + "    " + BPLR + " Filename : "  + xl5 )
    #Popen('filename.csv', shell=True)

	

	#show the image
	cv2.putText(image, "{}.png : {} : {:.2f}".format(xl5, text, laplac_variance2), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	key = cv2.waitKey(0)
