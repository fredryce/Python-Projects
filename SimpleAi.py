import random
from itertools import *
#game of sticks using probability basic idea of AI


class Player():
	def __init__(self):
		self.name = None

class Ai(Player):
	def __init__(self):
		super(Ai, self).__init__()
		self.name = ' '.join([chr(random.randrange(100)) for x in range(2)])
		self.array = {}
		


class GameofSticks():
	def __init__(self):
		self.startSticks = 20
		self.numSticks = self.startSticks
		self.welcome()
		self.chooseOption()

	def welcome(self):
		chose = "Rules: pick between 1-3 sticks, the player pick last stick loses the game\n"
		chose1 = "1. Choose simple player vs player\n2. Choose to train Ai and play against Ai\n3. Exit the game\n"
		while True:
			try:
				self.option = int(input(chose+chose1))
				if(self.option > 3 or self.option < 1):
					continue
				break
			except ValueError as e:
				print("please enter valid int..")
	def chooseOption(self):
		if self.option == 1:
			self.pvp()
		elif self.option == 2:
			value = 100
			while True:
				try:
					value = int(input("Round should Ai play against each other? "))
					if(value < 1):
						print("enter valid integer..")
						continue
					break
				except ValueError as e:
					print("enter valid integer..")


			self.train(value)
		else:
			exit()
	def winner(self, player):
		if self.numSticks <= 0:
			return True
		return False

	def pvp(self):
		player1 = Player()
		player2 = Player()
		setattr(player1, "name", str(input("what is player 1 name? ")))
		setattr(player2, "name", str(input("what is player 2 name? ")))
		players = [player1, player2]
		random.shuffle(players)
		print(getattr(players[0], "name") + " is going first")


		gen = cycle(players)
		currentPlayer = next(gen)
		while self.winner(currentPlayer) == False:
			try:
				picks = int(input("How many sticks you want to pick? "))
				self.numSticks -= picks
				print("{} picked {} sticks, There are {} sticks left..\n\n".format(getattr(currentPlayer, "name"), picks ,self.numSticks))
				currentPlayer = next(gen)
			except ValueError as e:
				print("Please choose between 1-3 sticks")

		print(getattr(currentPlayer, "name") + " won !!")


	def train(self, train_num):
		self.history = []
		
		for i in range(self.numSticks):
			startArray = [1,2,3]
			self.history.append(startArray)

		for i in range(train_num):
			ai1 = Ai()
			ai2 = Ai()
			winner = self.play(ai1, ai2)
			print("winner is: ", getattr(winner, "name"))
			for k, v in winner.array.items():
				self.history[int(k)].append(v)
			self.numSticks = self.startSticks
		rematch = True
		while rematch:

			newPlayer = Player()
			newPlayer.name = str(input("what's the player's name? "))
			newWinner = self.play(newPlayer, Ai(), train = False)
			print("winner is: ", getattr(newWinner, "name"))

			rematch = str(input("would you like to rematch? "))
			if rematch.upper() != "Y":
				rematch = False
			self.numSticks = self.startSticks


	def play(self, player1, player2, train = True):
		players = [player1, player2]
		random.shuffle(players)
		gen = cycle(players)
		currentPlayer = next(gen)
		while self.winner(currentPlayer) != True:
			try:
				if type(currentPlayer) == Player:
					picks = int(input("How many sticks you want to pick? \n"))
				else:
					picks = self.history[self.numSticks - 1][random.randrange(len(self.history[self.numSticks - 1]))]
					currentPlayer.array[str(self.numSticks - 1)] = picks
				self.numSticks -= picks
				if not train:
					print("{} picked {} sticks, There are {} sticks left..\n".format(getattr(currentPlayer, "name"), picks ,self.numSticks))
				currentPlayer = next(gen)
			except ValueError as e:
				print("Please choose between 1-3 sticks")
		return currentPlayer




if __name__ == "__main__":
	newgame = GameofSticks()
