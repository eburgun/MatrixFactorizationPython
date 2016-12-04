import math
import CSR
import time
import random
import numpy as np

class MFRecommender:
    
    def __init__(self, trainFile, testFile, epsilon, maxIters, kVal, lamb):
        self.__epsilon = epsilon
        self.__maxIters = maxIters
        self.__kVal = kVal
        self.__lamb = lamb
        self.__trainingData = CSR.CSR(trainFile)
        self.__trainingTranspose = CSR.CSR(trainFile)
        self.__trainingTranspose.transpose()
        self.__testingData = CSR.CSR(testFile)
        self.__createPandQWithRandom()
    
    def changeKVal(self,newK):
        self.__kVal = newK
        self.__createPandQWithRandom()
        
    def changeLamb(self,newLambda):
        self.__lamb = newLambda
        
    def trainSystem(self):
        i = 0
        fSumLast = 0
        while(i < self.__maxIters):
            self.__LS_GD(.0005)
            funcVal = self.__fFunction()
            if(i > 0 and math.sqrt((funcVal - fSumLast)**2)/fSumLast < self.__epsilon):
                break
            else:
                fSumLast = funcVal
            i +=1
                
    def __createPandQWithRandom(self):
        self.__pMatrix = np.random.random_sample((self.__trainingData.rows,self.__kVal))
        self.__qMatrix = np.random.random_sample((self.__trainingData.columns,self.__kVal))
        
        
    def __createPandQWithPredef(self,fillValue):
        self.__pMatrix = np.empty((self.__trainingData.rows,self.__kVal))
        self.__qMatrix = np.empty((self.__trainingData.columns,self.__kVal))
        self.__pMatrix.fill(fillValue)
        self.__qMatrix.fill(fillValue)
        
    def __fFunction(self):
        pNorm = self.__fNorm(self.__pMatrix)
        qNorm = self.__fNorm(self.__qMatrix)
        lambdaValue = (pNorm + qNorm) * self.__lamb
        fSum = 0
        for i in xrange(len(self.__pMatrix)):
            for j in xrange(len(self.__qMatrix)):
                fSum += (self.__trainingData.getElem(i,j) - np.dot(self.__pMatrix[i],self.__qMatrix[j].T))**2
        return fSum + lambdaValue
        
    def __fNorm(self,matrix):
        norm = 0
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[i])):
                norm += matrix[i][j] * matrix[i][j]
        return norm
    """
    def __LS_Closed(self, fixedMatrix, solvingMatrix,dataSet):
        lambdaI = np.identity(self.__kVal) * self.__lamb
        
        
        #YTY = np.dot(fixedMatrix.T, fixedMatrix)
        #YTY += lambdaI
        #YTYInv = np.linalg.inv(YTY)
        
        for i in xrange(len(solvingMatrix)):
            vectorByUser = np.zeros(shape = (self.__kVal))
            vectorByVector = np.zeros(shape = (self.__kVal,self.__kVal))
            for j in xrange(dataSet.row_ptr[i],dataSet.row_ptr[i+1]):
                vectorByUser += fixedMatrix[dataSet.column_idx[j]]*dataSet.rating[j]
                vectorByVector += np.multiply(fixedMatrix[dataSet.column_idx[j]].T,fixedMatrix[dataSet.column_idx[j]])
            vectorByVector += lambdaI
            
            vectorInv = np.linalg.inv(vectorByVector)
            solvingMatrix[i] = np.dot(vectorByUser, vectorInv)
    """
    def __LS_GD(self, learningRate):
        lambdaValue = 1 - (learningRate*2*self.__lamb)
        
        for i in xrange(len(self.__pMatrix)):
            sumMatrix = np.zeros(shape = (self.__kVal))
            for j in xrange(self.__trainingData.row_ptr[i],self.__trainingData.row_ptr[i+1]):
                sumMatrix += (self.__trainingData.rating[j] - np.dot(self.__pMatrix[i],self.__qMatrix[self.__trainingData.column_idx[j]].T))*self.__qMatrix[self.__trainingData.column_idx[j]]
            self.__pMatrix[i] = (lambdaValue * self.__pMatrix[i]) + (sumMatrix * learningRate * 2)
        
        for i in xrange(len(self.__qMatrix)):
            sumMatrix = np.zeros(shape = (self.__kVal))
            for j in xrange(self.__trainingTranspose.row_ptr[i],self.__trainingTranspose.row_ptr[i+1]):
                sumMatrix += (self.__trainingTranspose.rating[j] - np.dot(self.__pMatrix[self.__trainingTranspose.column_idx[j]],self.__qMatrix[i]))*self.__pMatrix[self.__trainingTranspose.column_idx[j]]
            self.__qMatrix[i] = (lambdaValue * self.__qMatrix[i]) + (sumMatrix*learningRate*2)
            
    def testMSERMSE(self):
        mse = 0.0
        for i in xrange(self.__testingData.rows):
            for j in xrange(self.__testingData.row_ptr[i],self.__testingData.row_ptr[i+1]):
                prediction = np.dot(self.__pMatrix[i],self.__qMatrix[self.__testingData.column_idx[j]].T)
                mse += (self.__testingData.rating[j] - prediction)**2
                
        mse /= self.__testingData.nonzero_values
        rmse = math.sqrt(mse)
        print "k = " + str(self.__kVal) + " lambda = " + str(self.__lamb) + " maxIters = " + str(self.__maxIters) + " epsilon = " + str(self.__epsilon) + " MSE = " + str(mse) + " RMSE = " + str(rmse)
        return mse, rmse
    
    def testingMethod(self):
        start = time.clock()
        testFile = open("results.txt","w")
        kVals = [10,50]
        lambVals = [0.01,0.1,1,10]
        iters = [50,100,200]
        epsilonVals = [0.0001,0.001,0.01]
        for i in xrange(len(kVals)):
            self.__kVal = kVals[i]
            
            for j in xrange(len(lambVals)):
                self.__lamb = lambVals[j]
                for k in xrange(len(iters)):
                    self.__maxIters = iters[k]
                    for l in xrange(len(epsilonVals)):
                        self.__epsilon = epsilonVals[l]
                        self.__createPandQWithRandom()
                        trainStart = time.clock()*1000
                        self.trainSystem()
                        trainFinish = (time.clock()*1000) - trainStart
                        testStart = time.clock()*1000
                        curMSE, curRMSE = self.testMSERMSE()
                        testFinish = (time.clock()*1000) - testStart
                        testFile.write(str(self.__kVal) + " " + str(self.__lamb) + " " + str(self.__maxIters) + " " + str(self.__epsilon) + " " + str(curMSE) + " " + str(curRMSE) + " " + str(trainFinish) + " " + str(testFinish) + "\n")
        
        testFile.close()
        print time.clock - start
        