

import sys
import MFRecommender
import time

trainingFile = sys.argv[1]
testFile = sys.argv[2]

running = True
kValue = 10
maxTries = 50
epsilon = 0.01
lambdaVal = 1
recommend = MFRecommender.MFRecommender(trainingFile, testFile, epsilon, maxTries, kValue, lambdaVal)


print "Data loaded."
print "Please choose from the options below:"
print "1. Define K Value. (Default == 3)"
print "2. Define lambda Value. (Default == 1)"
print "3. Train System"
print "4. Test System"
print "5. Create Test Report"
print "q. Exit"
while running:
    
    userInp = raw_input("What is your choice? ")
    
    if userInp == "1":
        while userInp == "1":
            KInput = raw_input("Please input desired K value. ")
            if KInput.isdigit() and KInput > 0:
                kValue = KInput
                recommend.changeKVal(int(kValue))
                userInp = 0
            else:
                print "That is not a valid number. Please try again"
            
    elif userInp == "2":
        while userInp == "2":
            lInput = raw_input("Please input desired N value. ")
            try:
                lambdaVal = float(lInput)
                recommend.changeLamb(float(lambdaVal))
                userInp = 0
            except ValueError:
                print "That is not a valid number. Please try again"
                
    elif userInp == "3":
        start = time.clock()
        recommend.trainSystem()
        print time.clock() - start
        userInp = 0
        
            
    elif userInp == "4":
        start = time.clock()
        recommend.testMSERMSE()
        print time.clock() - start
        userInp = 0
        
    elif userInp == "5":
        recommend.testingMethod()    
        userInp = 0
        
        
    elif userInp == "Q" or userInp == "q":
        running = False
        
    else:
        print "I'm sorry, I didn't quite catch that"
