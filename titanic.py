import autograd.numpy as np
import csv
from autograd import grad
from autograd.numpy import linalg as LA #used to calculate 1-norm for loop condition
import pickle
	


class MyLogisticReg:

	def __init__(self, options):
		self.ww0 = np.random.rand(8,1)

		pass


	def fit(self, X, y):
		"""Fit model, this function trains model parameters with input train data X and y
		X data array
		Y label vector"""
		xtrain = X
		ytrain = y

		"""condfun
		takes ww0 and ww0 prev
		and returns true if grad descent loop can continue 
		"""
		def condfun(ww0, ww0prev):
			important = (1/8) * LA.norm(ww0 - ww0prev, 1)
			#print (important) #for debugging threshold comparison
			return (important > 0.0005)

		"""desc
		takes ww0 and t 
		(t is unnecessary in this implementation, but could be used to calculate step size)
		and returns new ww0 based on gradient descent formula 
		"""
		def desc(ww0, t):
			alphat = .000001#1/(100*(1+t))
			ww0new = ww0 - alphat * grad_func(ww0)

			return ww0new


		"""loss
		takes ww0 (ww0 is vector w concatenated with w0 at the end)
		and calculates the loss 
		"""
		def loss(ww0):
			lamda = 1 #set lambda
			w = ww0[0:7] #split off w and w0
			w0 = ww0[7]
			eta = np.matmul(xtrain,w) + w0
			c1 = np.matmul(np.transpose(ytrain), eta)
			etabig = eta[eta > 30] #mask based on size of eta
			etasmall = eta[eta <= 30]
			etasmall = np.log( np.exp(etasmall) + 1 )

			c2 = np.sum(etabig) + np.sum(etasmall)
			loss = (lamda/2) * np.matmul(np.transpose(w),w)
			loss = loss - (c1-c2)
			return loss;


		"""loss2
		takes ww0 (ww0 is vector w concatenated with w0 at the end)
		and calculates the loss
		NOTE: This is the SGD varient of the loss function
		"""
		def loss2(ww0):
			#assign batches
			ind = int(t%49)
			xbatch = xbatches[ind]
			ybatch = ybatches[ind]

			#rest of function is identical to loss, barring the one commented line
			lamda = 1
			w = ww0[0:7]
			w0 = ww0[7]
			eta = np.matmul(xbatch,w) + w0 #vec
			c1 = np.matmul(np.transpose(ybatch), eta)
			etabig = eta[eta > 30]
			etasmall = eta[eta <= 30]
			etasmall = np.log( np.exp(etasmall) + 1 )

			c2 = np.sum(etabig) + np.sum(etasmall)
			loss = (lamda/2) * np.matmul(np.transpose(w),w)
			loss = loss - np.size(xtrain,0) * (c1-c2) / 10 #10 is the batch size
			return loss;


		#NOTE: UNCOMMENT THE BELOW IF RUNNING WITH SGD.
		#The below is used to divide the training set into batches of size 10
		"""
		ok = np.floor_divide(np.size(xtrain,0),10) * 10
		xbatches = xtrain[0:ok]
		xbatches = np.split(xbatches, 49)
		ybatches = ytrain[0:ok]
		ybatches = np.split(ybatches, 49)
		"""

		#NOTE: IF RUNNING SGD, change loss to loss2
		grad_func = grad(loss)

		#set initial values before beginning gradient descent
		ww0 = self.ww0
		ww0prev = self.ww0
		cond = True
		t = 0.0
		while(cond and t < 30000):
			
			#every 100 iterations, check the condition
			if (t%100 == 0 and t != 0):
				cond = condfun(ww0,ww0prev)
				#print( loss(ww0) ) #prints the loss. Useful for debugging

			ww0 = desc(ww0, t)
			if (t >= 100):
				ww0prev = desc(ww0prev, t - 100)

			t = t + 1.0

		#print("out of loop: " + str(t)) #debugging code to determine number of iterations to exit loop

		#update weights in MyLogisticReg
		self.ww0 = ww0
		pass


	def predict(self, X):
		"""Predict using the logistic regression model
		X is a data matrix"""
		
		realw = self.ww0[0:7] #split into w and w0
		realw0 = self.ww0[7]
		predlabels = np.matmul(X, realw) + realw0
		predlabels = predlabels > 0
		predlabels = predlabels * 1 #makes predlabels an int vector

		return predlabels



def evaluate(y_test, y_pred):
	"""Evaluate accuracy of predictions against true labels"""
	accuracy = (np.sum(np.equal(y_test, y_pred).astype(np.float))/y_test.size)
	return accuracy;





def main():

	#initialize X and y
	#NOTE: size of data is hard-coded
	X = np.zeros((710,7))
	y = np.zeros((710,1))

	##read in the data into X and y
	rownum = -1 #begin at -1, first row is useless

	with open('titanic_train.csv', newline ='') as f:
		reader = csv.reader(f)
		for row in reader:
			if (rownum == -1): #skip first row
				rownum = rownum+1
				continue

			#read into X	
			for i in range(7):
				X[rownum,i]= row[i+1]

			#read into y
			y[rownum]=row[0]

			#increment y
			rownum = rownum + 1
	#have X,y now

	#CODE TO SHUFFLE THE DATA. USED PRIMARILY FOR SGD
	"""
	Xy = np.append(X,y,1);
	np.random.shuffle(Xy)
	y= (Xy[:,7]).reshape((710,1))
	X = Xy[:,:-1]
	"""

	#split into 7:3
	xtrain = X[0:497]
	ytrain = y[0:497]

	xtest = X[497:710]
	ytest = y[497:710]
	#
	#
	#
	#data has been taken in and prepared


	"""run classifier on training data and 
	test on both training and test data
	options = None
	classy = MyLogisticReg(options)	
	classy.fit(xtrain,ytrain)
	predlabs = classy.predict(xtest)
	print( evaluate(predlabs, ytest) )
	print("TRAINNG SET:")
	pred2 = classy.predict(xtrain)
	print( evaluate(pred2, ytrain) )
	"""

	#train classifier, dump onto pickle, load pickle, and test
	options = None
	classifier = MyLogisticReg(options)
	classifier.fit(X,y)
	pickle.dump(classifier, open("titanic_classifier.pkl","wb") )
	infile = open("titanic_classifier.pkl","rb")
	ok = pickle.load(infile)
	pred = ok.predict(X)
	print( evaluate(pred, y) )
	


	#The following is code for K-folds cross validation, with k = 5
	"""
	options = None
	xfolds = np.split(X,5) #split data into k batches
	yfolds = np.split(y,5)
	
	#run k iterations of cross validation
	for k in range(0,5): 

		#initialize training and validation sets
		xtr = np.zeros((0,7))
		ytr = np.zeros((0,1))
		xval = np.zeros((0,7))
		yval = np.zeros((0,1))

		#accumulate training set and validation set
		for i in range(0,5):
			if (i == k):
				xval = xfolds[i]
				yval = yfolds[i]
			else:
				xtr = np.append(xtr, xfolds[i],0)
				ytr = np.append(ytr, yfolds[i],0)

		
		#run classifier on training and validation set and report accuracy
		classy = MyLogisticReg(options)
		classy.fit(xtr, ytr)
		predk = classy.predict(xval)
		print("k =: " + str(k))
		evaluate(predk, yval)
		
	"""




main()



