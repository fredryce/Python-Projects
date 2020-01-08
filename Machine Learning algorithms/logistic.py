#output is either 0 or 1 categorical variable result between 0 and 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LGSK
import tensorflow as tf
from sklearn.metrics import accuracy_score
class LogisticRegression(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.b_0, self.b_1 = self.estimate_coef()
		#self.plot_regression_line() 

	def estimate_coef(self): 
		# number of observations/points 
		n = np.size(self.x) 
	  
		# mean of x and y vector 
		m_x, m_y = np.mean(self.x), np.mean(self.y) 
	  
		# calculating cross-deviation and deviation about x 
		SS_xy = np.sum(self.y*self.x) - n*m_y*m_x 
		SS_xx = np.sum(self.x*self.x) - n*m_x*m_x 
	  
		# calculating regression coefficients 
		b_1 = SS_xy / SS_xx 
		b_0 = m_y - b_1*m_x

		print(b_0, b_1)
		return(b_0, b_1)


	def sigmoid(Z):
		return 1/(1+np.e**(-1))
	def loss(y, yhat):
		return np.sum(y*np.log(yhat) + (1-y)*(np.log(1-yhat)))/(-len(y))


	def plot_regression_line(self): 
		# plotting the actual points as scatter plot 
		plt.scatter(self.x, self.y, color = "m", 
				   marker = "o", s = 30) 
	  
		# predicted response vector 
		y_pred = self.b_0 + self.b_1*x 
	  
		# plotting the regression line 
		plt.plot(self.x, y_pred, color = "g") 
	  
		# putting labels 
		plt.xlabel('x') 
		plt.ylabel('y') 
	  
		# function to show plot 
		plt.show() 


class LogisticRegressionSK(LogisticRegression):
	def __init__(self, x, y):
		super(LogisticRegressionSK, self).__init__(x, y)
	def estimate_coef(self):
		reg = LGSK().fit(self.x[:, None], self.y[:, None]) #cant be 1d array, need to be [[0],[1],...] etc
		pred_output = reg.predict(self.x[:, None])
		accuracy = accuracy_score(self.y[:, None], pred_output)
		print(accuracy)
		print(reg.intercept_, reg.coef_)
		return reg.intercept_[0], reg.coef_[0]

class LogisticRegressionTensor(LogisticRegression):
	def __init__(self, x, y):
		super(LinearRegressionTensor, self).__init__(x, y)
		

	def estimate_coef(self):
		local_x, local_y =tf.placeholder('float'), tf.placeholder('float') 
		W = tf.Variable(np.random.randn(), name='W')
		b = tf.Variable(np.random.randn(), name='b')
		n = len(self.x)

		pred = tf.nn.sigmoid(tf.add(tf.multiply(local_x, W), b))
		cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.y)
		optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
		init = tf.global_variables_initializer()

		
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(1000):
				for (_x, _y) in zip(self.x, self.y):
					#print('Training %d'%epoch)
					sess.run(optimizer, feed_dict={local_x:_x, local_y:_y})
			print(sess.run(b), sess.run(W)) 
			return sess.run(b), sess.run(W)
				
		


if __name__ == "__main__":
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
	#LinearRegressionTensor(x,y)
	#LinearRegression(x,y)
	LogisticRegressionSK(x, y)