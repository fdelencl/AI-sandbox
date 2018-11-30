from memory_pipe.recorder import Recorder
from emulator import Emulator
from predictor.model import Predictor, device
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import numpy as np
import random
from gym.envs.classic_control.rendering import SimpleImageViewer

ram_show = SimpleImageViewer()
game = SimpleImageViewer()

predictor = Predictor().to(device)
model_path = './rom/predictor.model'
predictor.load(model_path)

learning_rate = 0.001
momentum = 0.001
batch_size = 1000

frame_prediction = 1
rewards = np.zeros((frame_prediction))
score = 0
index = 0

criterion = nn.L1Loss()
optimizer = optim.RMSprop(predictor.parameters(), lr=learning_rate, momentum=momentum)

actions = np.zeros((batch_size, 9))
inputs = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
predicted = torch.zeros((batch_size, 1), dtype=torch.float, requires_grad=True).to(device)
referenced = torch.zeros((batch_size, 1), dtype=torch.float, requires_grad=True).to(device)

reward = 0
epoch = 0
lives = 0
def optimize():
	global epoch, predicted, referenced, batch_size, actions
	epoch += 1
	predicted = predictor.estimate_value(inputs, actions)
	loss = criterion(predicted, referenced)
	print(epoch, loss.data)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	

class MyEmulator(Emulator):
	def __init__(self, fps=50, render=True, info=None):
		super(MyEmulator, self).__init__(fps=fps, render=render, info=info)
		self.batch = 0

	def before_step(self):
		self.user_actions = self.action_spectrum[index]

	def after_step(self, RAM, input, screen, info):
		global predicted, referenced, batch_size, index, actions, inputs, reward, lives, frame_prediction, rewards, score
		
		delta_score = info['score'] - score
		reward = 1 + delta_score
		rewards[self.batch % frame_prediction] = reward
		if lives != info['lives']:
			rewards = np.zeros((frame_prediction))
		score = info['score']
		lives = info['lives']
		if self.batch < batch_size:
			# inp = torch.Tensor([list(RAM[0:256] + RAM[768:2048])]).to(device)
			inp = []
			for key, value in info.items():
				inp.append(value)
			inp = torch.Tensor([inp])

			inputs[self.batch] = inp[0]
			inpu = torch.zeros((len(self.action_spectrum), 79), dtype=torch.float, requires_grad=True).to(device)

			for i in range(0, len(self.action_spectrum)):
				inpu[i] = inp[0]
			prediction = predictor.estimate_value(inpu, self.action_spectrum)
			# print(prediction)
			global epoch
			if epoch < 0:
				index = random.randint(0, 9)
			else:
				index = 0
				indexes = []
				for i in range(0, len(self.action_spectrum)):
					if prediction[i] > prediction[index]:
						index = i
						indexes = [index]
					if prediction[i] == prediction[index]:
						indexes.append(i)
				index = random.choice(indexes)
			# if len(indexes) != len(self.action_spectrum):
			# 	print(len(indexes))
			actions[self.batch] = self.action_spectrum[index]
		if self.batch >= frame_prediction:
			referenced[self.batch - frame_prediction] = sum(rewards)
		
		if self.batch >= batch_size - 1 + frame_prediction:
			predictor.train()
			optimize()
			predictor.eval()

			inputs = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
			predicted = torch.zeros((batch_size, 1), dtype=torch.float, requires_grad=True).to(device)
			referenced = torch.zeros((batch_size, 1), dtype=torch.float, requires_grad=True).to(device)
			self.batch = 0
		else:
			self.batch += 1
		super(MyEmulator, self).after_step(RAM, input, screen, info)

	def done(self):
		global model_path
		predictor.save(model_path)
		super(MyEmulator, self).done()

emul = MyEmulator(fps=0, render=True, info='./rom/data.json')
emul.run()
sys.exit(0)