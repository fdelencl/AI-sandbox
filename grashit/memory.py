import random
import pickle

def Transition(a, b, c, d):
	return [a, b, c, d]

class Memory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

	def reset():
		self.memory = []
		self.position = 0

	def save(self, filename):
		filename = filename + '.kitty'
		f = open(filename, 'w+b')
		pickle.dump(self.memory, f)
		f.close()

	def load(self, filename):
		f = open('loli.kitty', 'r+b')
		mem = pickle.load(f)
