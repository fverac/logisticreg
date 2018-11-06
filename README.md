# logisticreg
Logistic Regression classifiers for 2 different datasets.

Fabio Vera

** This project was built and run using python3 **

to run:

python3 [filename]

There are two different files/classifiers for the different datasets
(named titanic.py and mnist.py)

In both of them, Full gradient descent is used to minimize the loss function by default,  with Stochastic Gradient Descent (SGD) being commented out.
In order to run SGD, Follow the instructions given in the comments at the top
of the files.

Code for K fold cross validation is left in, but commented out

Code for timing experiments is left in as well, but commented out

To change lambda for experiments, go to loss or loss2 functions in the
MyLogisticReg class and change lambda there


Final Accuracy estimations after K-fold cross validation are as follows:
Titanic dataset classifier: 0.760 +/- 0.036
MNIST data set classifier: 0.9553 +/- 0.0067


