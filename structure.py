#python sets
import random
import sqlite3

setA = set((10,30,13,10))
setB = set((random.randrange(100) for x in range(100)))
print(setB)

#double underscore methods

class poly():
	def __init__(self, *coeff): #*coeff means that it takes in multiple args
		self.coeff = coeff
	def __repr__(self):
		return "Poly({})".format(self.coeff)

	def __add__(self, other): #other is another poly class that it takes in to add
		return poly(*(x + y for x, y in zip(self.coeff, other.coeff))) #adding the star will take the generator out of the generator

p1 = poly(1,2,3)
p2 = poly(4,5,6)

print(p1) #used by repr double underscore methods
print(p1 + p2)


print(*zip(p1.coeff, p2.coeff))



#metaclasses extends type
class Base(): #libary
	def foo(self):
		return "foo"

class Derived(Base): #user
	def bar(self):
		return self.foo()

#derived class will fail if the base class methods is missing
#how to change this? meta classes!!!


class BaseMeta(type):
	def __new__(klass, name, parent, body):
		if not 'bar' in body:
			print("function has no bar") 

		print("BaseMeta().__new__ out", body)
		return super().__new__(klass, name, parent, body)


class Base1(metaclass = BaseMeta): #libary
	def foo(self):
		return self.bar()	#where does it get self.bar from?

class Derived1(Base1): #user
	def bar(self):
		return "bar"	


mybar = Derived1()
mybar.bar()



#power of decorators

import time

def timer(function):
	def innnerWrapper(*args, **kwargs):
		start = time.time()
		value = function(*args, **kwargs)
		total = time.time() - start

		print("time took", total)

		return value #has to return this, then this return value is carried to the return value of the function its decorating
	return innnerWrapper

@timer # have the timer function defined once, takes in any function and perform the task. used to add on more complex stuff along the projects
def add(x,y):
	return x + y
@timer
def hello(x,y,z):
	for i in range(100):
		x = x + y + z
	return "hello" + str(x + y + z)
@timer
def flexAdd(*args, **kwargs):  #having *args and *key word arguments allows the flexible pass in of parameters
	print(kwargs["myvalue"])
	return sum(args)


print("3 + 2 is : ", add(3,2))
print("hello function: ", hello(1,34,5))
print(flexAdd(3,5,6,5,6,7,77,4,3,2,5, myvalue="someday"))

#generators
#when gernerating a list, you dont want to wait for the entire list to finish generating before you can use it.
#generator returns a value as a time as it generates more

def compute():
	for i in range(10):
		time.sleep(0.5)
		yield i
for value in compute():
	print(value)

#another use of generators coroutines(generator) vs subroutines. coroutines + multi thread. coroutines gives permission and yeilds, and the mission is done by multiple threads
#for webscrping, you want to be able to grab one page at a time using yeild, then give that one web page to one of the threads to handle

#context manager/ resource allocation. match multple actions togther, such as opening a file must end with closing a file
