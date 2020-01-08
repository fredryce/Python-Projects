import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LRSK
import tensorflow as tf
class LinearRegression(object):
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


class LinearRegressionSK(LinearRegression):
	def __init__(self, x, y):
		super(LinearRegressionSK, self).__init__(x, y)
	def estimate_coef(self):
		reg = LRSK().fit(self.x[:, None], self.y[:, None]) #cant be 1d array, need to be [[0],[1],...] etc
		print(reg.intercept_, reg.coef_)
		return reg.intercept_[0], reg.coef_[0]

class LinearRegressionTensor(LinearRegression):  #using gradient decent is good when the data size is big
	def __init__(self, x, y):
		super(LinearRegressionTensor, self).__init__(x, y)
		

	def estimate_coef(self):
		local_x, local_y =tf.placeholder('float'), tf.placeholder('float') 
		W = tf.Variable(np.random.randn(), name='W')
		b = tf.Variable(np.random.randn(), name='b')
		n = len(self.x)

		pred = tf.add(tf.multiply(local_x, W), b)
		cost = tf.reduce_sum(tf.square(pred- local_y))/(n)
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
	LinearRegressionTensor(x,y)
	LinearRegression(x,y)
	LinearRegressionSK(x, y)

