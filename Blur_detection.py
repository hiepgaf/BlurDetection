# USAGE
# python Blur_detector.py --images sampleImages
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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=3500.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

try:
	os.remove('Blur_detector_results.csv')
except OSError:
    pass
open_file = open("Blur_detector_results.csv", "a")
columnTitleRow = "BPLR_Name,BlurType,Laplacian Variance, FFT_Mean, FFT_Freq, ImageFileNumber, Imagetype\n"
open_file.write(columnTitleRow)

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

def fourier_transform(image):
        np.seterr(all = 'ignore')
        img_gry = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rows,cols = img_gry.shape
        crow, ccol = rows/2, cols/2
        f = np.fft.fft2(img_gry)
        fshift = np.fft.fftshift(f)
        fshift[crow-75:crow+75,ccol-75:ccol+75] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_fft = np.fft.ifft2(f_ishift)
        img_fft = 20*np.log(np.abs(img_fft))
        freqs = np.fft.fftfreq(len(img_gry))
        w = np.fft.fft(fshift)
        idx = np.argmax(np.abs(w))
        freq = freqs[idx]
        freq_in_hertz = abs(freq * 10)
        fourier_transform_mean = np.mean(img_fft)
        return(freq_in_hertz, fourier_transform_mean)


for imagePath in paths.list_images(args["images"]):

	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	#print(imagePath)
	image = cv2.imread(imagePath)
	laplac_variance = w2d(image,'db1',7)
	laplac_variance2 = float(laplac_variance)
	(FFT_Mean,FFT_Freq) = fourier_transform(image)
	#print (FFT_Mean)
	
	xl = str(imagePath)
	(xl1,xl2) = xl.split("/",1)
	(xl3,xl4) = xl2.split("image",1)
	(xl5,xl6) = xl4.split(".",1)
	BPLR = "R1"
	#imagenumber = 30
	if int(xl5) <=30:
		BPLR = "R3_NON_BLUR_LENS_Tester_Image"
	elif int(xl5) >30 and int(xl5) <= 60:
		BPLR ="R1_BLUR_LENS_Tester_Image"
	elif int(xl5) >60 and int(xl5) <= 90:
		BPLR = "R2_NON_BLUR_LENS_Tester_Image"
	elif int(xl5) >90 and int(xl5) <= 120:
		BPLR = "L1_NON_BLUR_LENS_Tester_Image"
	elif int(xl5) >120 and int(xl5) <= 150:
		BPLR = "L2_NON_BLUR_LENS_Tester_Image"
	elif int(xl5) >150 and int(xl5) <= 180:
		BPLR = "L3_NON_BLUR_LENS_Tester_Image"
	elif int(xl5) >180 and int(xl5) <= 190:
		BPLR = "R2_NON_BLUR_LENS__Sharper_Image"
	elif int(xl5) >190 and int(xl5) <= 200:
		BPLR = "R1_BLUR_LENS__Sharper_Image"
	else:
		BPLR = "confused state" #thats a joke

	text = "Not-Blurry"
	Imagetype = "Tester_Image"


	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if int(xl5) < 180:
		Imagetype = "Tester_Image"
		if laplac_variance > 3500.0 and FFT_Freq > 4.0 and FFT_Mean < 71.0:
			text = "Blurry"

	elif int(xl5) >180:
		Imagetype = "Sharper_Image"
		if laplac_variance < 5000.0 and FFT_Freq > 0.5:
			text = "Blurry"
	else:
		text = "Not sure bruh!"

	
	filewriter = csv.writer(open_file)
	filewriter.writerow([BPLR, text, laplac_variance, FFT_Freq ,FFT_Mean, str(xl5), Imagetype])
	#print(xl4 + " " + xm + "  " +  text)
	print("Image Quality : "+ text + " Laplacian Variance: " + str(laplac_variance2) +" Type of BPLR : " + BPLR +" FFT_Freq : " + str(float(FFT_Freq)) + " FFT_Mean :" + str(float(FFT_Mean)) + " File Name : "  + xl5 + " Imagetype : " + Imagetype )
    #Popen('Blur_detector_results.csv', shell=True)

	

	#show the image
	#cv2.putText(image, "{}.png : {} : {:.1f}".format(xl5, text, laplac_variance2), (10, 30),
	#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	#cv2.imshow("Image", image)
	#key = cv2.waitKey(0)
