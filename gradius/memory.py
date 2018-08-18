from collections import namedtuple
import random
import numpy as np

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
		flat_format = []
		for setup in self.memory:
			for i in setup:
				flat_format.append(i)
		byte_format = bytearray(flat_format)
		f.write(byte_format)
		f.close()
