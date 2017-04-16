#!/usr/bin/env python
import rospy
import struct
import ctypes
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import random
from sensor_msgs.msg import PointCloud2

#################Program Introduction ####################################
#Created by Julia Szeles and Noel Tay 2017.02.22
#In this program, we subscribe to a PointCloud2 in RVIZ simulator
#we store the PointCloud2 x,y,c,z coordinates in an array
#We only consider points which x coordinate is greater than 1.5m and the y is smaller than 0 (to increase the efficency)
# 1-In the RANSAC part, we randomly select 200 data points and calculate the cita values (1x3 vector) with the Linear regression
#2-From the original array we clear the randomly selected elements, and with the previously calculated cita values we calculate the error
#With the RANSAC, we do the step 1 and 2 for 100 times and we update the error value every time when the newly calculated error #value is smaller then the previous one, that will be the best error value
#We also store the cita vector of the best error value
#####################Program introduction end ############################

#Store the PointCloud2 elements into a file for further checking !Do not use it in the actual program makes it slow
file = open('workfile.txt','w')

#Variables for getting the PointCloud2 data
subscriber = 1 			#We must subscribe for the PointCloud2 once
x_array = [] 			#Store the PointCloud2 elements (x,y,c)
y_array = [] 			#Store the PointCloud2 elemets (z)
arrayWithXYCZ = [] 		#Merge the x_array and the y_array into one big array
#For using the columns as vectors in the linear regression and the error calculating
firstColumnArrayX = [] 		#x column vector
secondColumnArrayY = [] 	# y column vector
thirdColumnArrayConstant = [] 	# constant column (will be only 1 elements)

#############testing part #########
arrayForTestingTwoDimensional = []
testingXColumn = []
testingYColumn = []
testingCColumn = []
arrayForZAlliasY = []

arrayForTestingTwoDimensionalSmall = []
testingXColumnSmall = []
testingYColumnSmall = []
testingCColumnSmall = []
arrayForZAlliasYSmall = []

wholeBigAssTestArray = []

########testing part ##########

#In this function, we subscribe to the PointCloud2 and insert the data to the right arrays
#From the msg variable, the function gets out the x,y and z coordinates, the constant is created by us (only 1-s)
#If it is finnished we call the applyRansac(array) function for the arrayWithXYCZ array
def callback(msg):
	#print(msg)
	data_out = pc2.read_points(msg,skip_nans=True)
	while subscriber == 1:
		try:
			int_data = next(data_out)

			s = struct.pack('>f',int_data[3])
			i = struct.unpack('>l',s)[0]
			pack = ctypes.c_uint32(i).value

			r = (pack & 0x00FF0000)>> 16
			g = (pack & 0x0000FF00)>> 8
			b = (pack & 0x000000FF)

			#create matrix from PointCloud2 data to the xycz matrix and also to the column vectores for x,y,c and z
			#increase efficency by only adding x which are bigger than 1.5 and y which are smaller than 0
			#if the x is bigger than 1.5, that means that the table must be taller then my heaf
			if int_data[0] < 1.5 and int_data[1] < 0:
				x_array.append([int_data[0],int_data[1],1])
				firstColumnArrayX.append(int_data[0])
				secondColumnArrayY.append(int_data[1])
				thirdColumnArrayConstant.append(1)

				y_array.append(int_data[2])
				file.write(str(int_data[0])+","+str(int_data[1])+","+str(1)+"\n")
				arrayWithXYCZ.append([int_data[0],int_data[1],1,int_data[2]])
		
		#Program comes here if we run out of data points from the PointCloud2
		#to get the best error value we need to store the previous one, we use global variables for this
		except Exception as e:
			rospy.loginfo(e.message)
			setErrorValueToStarterValue(5) 	#global variable initializing
			testPrintErrorValueOut() 	#check if it works by printing out the current value
			if subscriber == 1: 		#only call this function once
				applyRansac(arrayWithXYCZ)
			setGlobalVariableToZerro() 	#set subscriber to 0 in order to not run the function again
			test() 				#check if it is 0 by printing it out
			file.flush
			file.close

#This function is called in the main, subscribes to the point cloud till there are points,rospy.spin is needed
def listener():
	rospy.init_node('writeCloudsToFile', anonymous = True)#give a name to the file we are going to write the data
	rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/points",PointCloud2, callback) #subsrcibe to the PointCloud2
	rospy.spin()

#Linear Regression calculation function, getting the cita value
#In this function with the RANSAC we randomly select 200 points and calculate the cita value of it 
#We are using the following equation: cita = ((XArray(trans)*XArray)^-1)*XArray(trans)*Y(ZColumn)
def equations(array,arrayY,xColumn,YColumn,CColumn,ZColumn):
	
	transposeOfTheArray = zip(*array) #get the transpose of the XArray
	array = np.array(array).astype('float') #cast it to be a float (Python requires it)
	transposeOfTheArray = np.array(transposeOfTheArray).astype('float')
	
	multiplicationOfTransposeAndNormal = np.dot(transposeOfTheArray,array)#multiply the 2 arrays XArray(trans)*XArray
	inverse = np.linalg.inv(multiplicationOfTransposeAndNormal)#inverse(XArray(trans)*XArray)
	inverse = np.array(inverse).astype('float')
	multiplyInverseAndTranspose = np.dot(inverse,transposeOfTheArray)#inverse(XArray(trans)*XArray) * XArray(trans)

	count = sum(len(x) for x in multiplyInverseAndTranspose) #check how many elements did we get

	# multiply the multiplyInverseAndTranspose with the y_array(ZColumn) to get the cita
	y_newArray = np.array(arrayY).astype('float')
	cita_array = np.dot(multiplyInverseAndTranspose,y_newArray) #get the 1x3 cita vector
	print(cita_array)

	#Calculate the error of the big array without the random selected items
	getTheErrorValue(cita_array,y_newArray,xColumn,YColumn,CColumn,ZColumn)

#In this function we calculate the best error value by using the following equations
#X[]*Cita1 + Y[]*Cita2 + C[]*Cita3 = Z
#We recieved the cita value by using the randomly selected items, we apply the cita value for the whole array without the #randomly selected items (all done in the applyRansac(array) function)
def getTheErrorValue(citaArray,yArray,XCol,YCol,CCol,ZCol):
	testPrintErrorValueOut()#global variable error value has been created and set to 5 (infinite) is the best

	#multiply X[] with cita1
	xColumnMultipliedWithCita1 = map(lambda x: x * citaArray[0], XCol) #firstColumnArrayX
	countX = len(xColumnMultipliedWithCita1)
	
	#multiply Y[] with x cita2
	yColumnMultipliedWithCita2 = map(lambda y: y * citaArray[1], YCol)
	countY = len(yColumnMultipliedWithCita2)

	#multiply C[] cita 3 (which is multiplying cita3 with 1 so it is gonna be cita3)
	cColumnMultipliedWithCita3 = map(lambda c: c * citaArray[2], CCol)

	#The above created array should be summed and we store the sums in the resultArray[i]
	#xColumnMultipliedWithCita1[i] + yColumnMultipliedWithCita2[i] + cColumnMultipliedWithCita3[i] = Z
	#While loop goes till we reach the end of the xColumnMultipliedWithCita1[]
	resultArray = []	
	i = 0	
	while i < countX:
		addition = xColumnMultipliedWithCita1[i] + yColumnMultipliedWithCita2[i] + cColumnMultipliedWithCita3[i]
		resultArray.append(addition)
		i += 1
	

	#To get the error value we substract the Z[j] from resultArray[j]
	#We loop till the end of the resultArray and calculate the error for every row
	#We cannot allow to have a negative error value so we only consider the absolut value of the substraction
	#We then store the calculated elements into an errorResult[]
	countResult = len(resultArray)	
	j = 0
	errorResult = []
	while j < countResult:
		substraction =  resultArray[j] - ZCol[j]
		if substraction < 0:
			substraction = substraction * -1
		errorResult.append(substraction)
		j += 1

	#To get the error value for the whole array, we need to calculate the avarage value of the errorResult[]
	#errorResult[]sumOfElements / errorResultCount (in case if you dont know how to calculate the avarage) 
	errorResultCount = len(errorResult) #counter for errorResult
	sumOfError = 0 #sum the errorResult items and store it to the sumOfError
	for errorItem in errorResult:
		sumOfError += errorItem
	
	avarageError = sumOfError / errorResultCount
	print("The avarage error is: ",avarageError)
	
	#Since we are using RANSAC to get the best error value, we compare the newly calculated avarageError to the old one
	#which was stored as a global variable and if the new error value is smaller than the previous
	#Update the  previously stored error value with the new one
	#We also store the cita_array[] which belongs to the new best error value
	if avarageError < bestErrorValue:
		print("\n")
		print("Best value has changed")
		setLongTermCitaArray(citaArray)
		setErrorValueToStarterValue(avarageError)
		changeBestValue(avarageError)
		testPrintErrorValueOut()
		print("The cita value has changed")
		printOutLongTermCitaArray()
		print("\n")

###################Global Variable set up START ######################
#Python defines local and global variables, we need to initalize and update global variables only in functions
#I guess because python wants you to use global ones ONLY in the condition if you know what you are doing	
def setGlobalVariableToZerro():
	global subscrbalVariables()
	iber
	subscriber = 0

def setErrorValueToStarterValue(v):
	global bestErrorValue
	bestErrorValue = v

def setLongTermCitaArray(value):
	global longTermCitaArray
	longTermCitaArray = value

def printOutLongTermCitaArray():
	print("The value of the cita array is: ",longTermCitaArray)

def testPrintErrorValueOut():
	print("The error value is: ",bestErrorValue)

def test():
	print("Subscriber is",subscriber)

def changeGlobalVariable():
	subscriber = 0
	print("Subscriber has changed",subscriber)

def changeBestValue(value):
	bestErrorValue = value
	print("Best error value has changed: ",bestErrorValue)
###################Global Variable set up ENDING ######################

#Testing part, generating multidimensional array with random numbers
def tryIfMyEquationWorksGenerateArrayWithRandomNumbers(rangeNum, twoDimensionalArray,testXColumn,testYColumn,testCColumn):
	
	i = 0
	while i < rangeNum:
		x = random.uniform(0.1,3.9)
		y = random.uniform(0.1,3.9)
		c = 1
		twoDimensionalArray.append([x,y,c])
		testXColumn.append(x)
		testYColumn.append(y)
		testCColumn.append(c)
		i += 1
	count = sum(len(h) for h in twoDimensionalArray)
	print("Number of elements in 3X20 test array",count)
	#print(twoDimensionalArray)

def calculateTheZForTesting(xCol,YCol,CCol,cite1,cite2,cite3,resultZArray):
	count = len(xCol)
	
	i = 0
	while i < count:
		zResult = xCol[i] * cite1 + YCol[i] * cite2 + CCol[i] * cite3 
		randNumForZNoise = random.uniform(0.001,0.03)
		zResult += 0 #randNumForZNoise
		resultZArray.append(zResult)
		i += 1

	countZArray = len(resultZArray)
	print("Number of items in the calculatedZ: ",countZArray)
	#print(resultZArray)

#RANSAC part where we select 200 points randomly, calculate the cita for these 200 points
#Remove the 200 randomly selected points from the original two dimensional array
#For the remaining array with the calculated cita values calculate the avarage error
#If the avarage error is better than the previous avarage error update the error value
#We run the above described process 10 times 
def applyRansac(wholeArray):
	
	testPrintErrorValueOut()
	counterForGettingTheError = 0
	while counterForGettingTheError < 10:
		#select random points from the pointCloud
		secore_random = random.SystemRandom()
		randomlySelectedElements = []
		remainingElements = []	
		remainingArray = []

		testArray = np.array([['1','2','3'],['4','5','6'],['7','8','9'],['11','12','13']])
		testArray2 = np.array([['14','1','3'],['4','5','6'],['7','8','9'],['11','12','13']])
		
		#randomly select elemets from the original array
		i = 0
		while i < 200:
			item = secore_random.choice(wholeArray)
			randomlySelectedElements.append(item)
			i += 1
				
		#delete the randomly selected elements from the original array
		remainingArray = finalTest(wholeArray,randomlySelectedElements)
		countRemainingArray = sum(len(h) for h in remainingArray)
		print("Remaining array counter",countRemainingArray)

		xColumnArrayBig = returnColumnOfTheMatrix(remainingArray,0) 
		yColumnArrayBig = returnColumnOfTheMatrix(remainingArray,1) 
		cColumnArrayBig = returnColumnOfTheMatrix(remainingArray,2) 
		zOrYColumnArrayBig = returnColumnOfTheMatrix(remainingArray,3) 

		zForTheRandomlySelectedArray = returnColumnOfTheMatrix(randomlySelectedElements,3)
	
		countX = len(xColumnArrayBig)
		print("X column array:",countX)

		#Getting only the elements from the randomly selected items which are the X,Y,C
		arrayForX = np.delete(randomlySelectedElements,3,axis=1)
	
		#calculating the cita with the following function
		equations(arrayForX,zForTheRandomlySelectedArray,xColumnArrayBig,yColumnArrayBig,cColumnArrayBig,zOrYColumnArrayBig)
		
		counterForGettingTheError += 1
	
	#display the best error value after the loop is over
	print("\n")
	print("We recieved the final sample")
	testPrintErrorValueOut()
	printOutLongTermCitaArray()

def returnColumnOfTheMatrix(matrix, i):
		return [row[i] for row in matrix]	

def finalTest(arrayOrig,randomizedArray): #arrayOrig,randomizedArray
	testArray = np.array([['1','2'],['3','12'],['11','22'],['77','88']])
	testArray2 = np.array([['1','2'],['5','12'],['11','22']])

	testArraySet = set([tuple(x) for x in arrayOrig])
	testArraySet2 = set([tuple(y) for y in randomizedArray])
	
	#itemsToRemove = set(testArray2)
	remainingElements = filter(lambda x: x not in testArraySet2, testArraySet)
	#print(remainingElements)	
	return remainingElements

def mergeTestArrayAndZColumnTogother(array,zCol,isBig): #array,zCol
	testArray = np.array([['1','2'],['3','12'],['11','22'],['77','88']]) #4X2
	testVector = np.array(['2','15','22','88']) #4x1
	#testArray = np.append(testArray,testVector, 1)
	testArray = np.column_stack([array,zCol])
	
	if isBig == 0: #Big Array
		arrayForTestingTwoDimensional = testArray
		print(1)
	if isBig == 1:
		arrayForTestingTwoDimensionalSmall = testArray
		print(2)

	#print("Test array \n")
	#print(testArray)
	return testArray

def mergeTwoMultiDimensionalArray(arrayBig,arraySmall): #
		
	countB = sum(len(h) for h in arrayBig)	
	countS = sum(len(h) for h in arraySmall)

	print("Counting Big array", countB)
	print("Counting Small array",countS)

	testArray = np.array([['1','2'],['3','12'],['11','22'],['77','88']]) 
	testArray2 = np.array([['11','22'],['33','122'],['111','222'],['777','888']])
	testArray = np.vstack([arrayBig,arraySmall])

	count = sum(len(h) for h in testArray)

	#print(testArray)
	#print("Count for testing the array:",count)
	return testArray

if __name__ == '__main__':
	#globalVariables()
	listener()
	#tryIfMyEquationWorksGenerateArrayWithRandomNumbers()
	#calculateTheZForTesting()
	#equations(arrayForTestingTwoDimensional,arrayForZAlliasY)
	#applyRansac(arrayWithXYCZ)

	#testArray = np.array([['1','2'],['3','12'],['11','22'],['77','88']])
	#testArray2 = np.array([['1','2'],['5','12'],['11','22']])
	#finalTest(testArray,testArray2)

	#setErrorValueToStarterValue(5)
	#testPrintErrorValueOut()

	#Big data testing creating the two arrays and marging them togother
	#tryIfMyEquationWorksGenerateArrayWithRandomNumbers(50, arrayForTestingTwoDimensional,testingXColumn,testingYColumn,testingCColumn)
	#calculateTheZForTesting(testingXColumn,testingYColumn,testingCColumn,1.9,2,3,arrayForZAlliasY)
	#arrayForTestingTwoDimensional = mergeTestArrayAndZColumnTogother(arrayForTestingTwoDimensional,arrayForZAlliasY,0)
	#tryIfMyEquationWorksGenerateArrayWithRandomNumbers(10, arrayForTestingTwoDimensionalSmall,testingXColumnSmall,testingYColumnSmall,testingCColumnSmall)
	#calculateTheZForTesting(testingXColumnSmall,testingYColumnSmall,testingCColumnSmall,1,2,10,arrayForZAlliasYSmall)
	#arrayForTestingTwoDimensionalSmall = mergeTestArrayAndZColumnTogother(arrayForTestingTwoDimensionalSmall,arrayForZAlliasYSmall,1)

	#wholeBigAssTestArray = mergeTwoMultiDimensionalArray(arrayForTestingTwoDimensional,arrayForTestingTwoDimensionalSmall)

	#Apply RANSAC
	#applyRansac(wholeBigAssTestArray)
