from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
	group = array(([1.0,1.1],[1.0,1.1],[0,0],[0,0.1]))
	labels = ['A','A','B','B']
	return group,labels
	

		
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector		
		

##归一化特征值
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normalDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normalDataSet = dataSet - tile(minVals,(m,1))
	normalDataSet = dataSet/tile(ranges,(m,1))
	return normalDataSet,ranges,minVals

	
#测试算法
def datingClassTest():
	hoRatio = 0.10
	datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
	normMat,ranges,minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],morMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print('the classfier came back with :%d,the real answer is :%d' % (classifierResult,datingLabels[i]))
		if (classifierResult != datingLabels[i]):
			errorCount +=1.0
	print('the total error rate if %f' % (errorCount/float(numTestVecs)))
		
	
def classifyPerson():
	resultList = ['not at all',['in samll doses'],['in range doses']]
	percentTats = float(input('percentage of time spent playing video games?'))
	ffMiles = float(input('frequent filer mile earned per year?'))	
	iceCream = float(input('liter od icecream consumed per year?'))
	
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat,ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles,percentTats,iceCream])
	classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print('you will like the person:',resultList[classifierResult - 1])
	


##手写识别系统

#将图像转化为测试向量
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		linestr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(linestr[j])
	return returnVect
	

##手写数字识别系统测试
def handwritingClassTest():
	#获取目录内容
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		#从文件名解析分类数字
		fileNameStr = trainingFileList[i]
		fileStr1 = fileNameStr.split('.')[0]
		classNumStr = int(fileStr1.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	
	testFileList = listdir('testDigits')
	errorCount = 0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr2 = fileNameStr.split('.')[0]
		classNumStr = int(fileStr2.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		
		print('the classifier came back with %d the real answer is %d' % (classifierResult,classNumStr))
		if (classifierResult != classNumStr):
			errorCount +=1
	print('\nthe total error is %d' % errorCount)
	print('\nthe error rate is %f' % (errorCount/float(mTest)))
	
		
	
	
		
		
	
	























