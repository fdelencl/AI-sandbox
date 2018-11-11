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
from gym.envs.classic_control.rendering import SimpleImageViewer

ram_show = SimpleImageViewer()
game = SimpleImageViewer()

predictor = Predictor().to(device)
model_path = './rom/predictor.model'
predictor.load(model_path)

learning_rate = 0.0005
momentum = 0.001
batch_size = 1000

frame_prediction = 5

criterion = nn.L1Loss()
optimizer = optim.RMSprop(predictor.parameters(), lr=learning_rate, momentum=momentum)

predicted = torch.zeros((batch_size, 1), dtype=torch.float, requires_grad=True).to(device)
referenced = torch.zeros((batch_size, 1), dtype=torch.float, requires_grad=True).to(device)

hidden = torch.zeros((1, 79), dtype=torch.float, requires_grad=False).to(device)

epoch = 0
def optimize():
	global epoch
	epoch += 1
	print(predicted.shape)
	print(referenced.shape)
	loss = criterion(predicted, referenced)
	print(epoch, loss.data)
	optimizer.zero_grad()
	loss.backward()
	print("epoch1 done")
	optimizer.step()
	

class MyEmulator(Emulator):
	def __init__(self, fps=50, render=True, info=None):
		super(MyEmulator, self).__init__(fps=fps, render=render, info=info)
		self.batch = 0

	def before_step(self):
		self.user_actions = self.env.action_space.sample()


	def after_step(self, RAM, input, screen, info):
		global predicted, referenced, batch_size, hidden
		reference = info['score']

		if self.batch < batch_size:
			inp = []
			for key, value in info.items():
				inp.append(value)
			inp = torch.Tensor([inp]).to(device)
			prediction = predictor.estimate_value(inp, [self.user_actions])
			predicted[self.batch] = prediction;
		if self.batch >= frame_prediction:
			referenced[self.batch - frame_prediction] = reference
		
		if self.batch >= batch_size - 1 + frame_prediction:
			print(self.batch)
			optimize()
			hidden = torch.zeros((1, 79), dtype=torch.float, requires_grad=False).to(device)
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

emul = MyEmulator(fps=0, render=False, info='./rom/data.json')
emul.run()
sys.exit(0)