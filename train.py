from memory_pipe.recorder import Recorder
from emulator import Emulator
from screen_reader.model import ScreenReader, device
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from gym.envs.classic_control.rendering import SimpleImageViewer

ram_show = SimpleImageViewer()

screen_reader = ScreenReader().to(device)
screen_reader.load()

learning_rate = 0.0015
momentum = 0.1
test_every = 100000
test_length = 1000

criterion = nn.L1Loss()
optimizer = optim.RMSprop(screen_reader.parameters(), lr=learning_rate, momentum=momentum)

test_result = 0

def optimize(prediction, reference):
	screen_reader.zero_grad()
	loss = criterion(prediction, reference)
	loss.backward()
	optimizer.step()

class MyEmulator(Emulator):
	def __init__(self, fps=50, render=True):
		super(MyEmulator, self).__init__(fps=fps, render=render)
		self.test_inputs = [self.env.action_space.sample() for i in range(0, test_length)]
		self.testing = True
		self.frame = 0
		print(len(self.test_inputs))

	def before_step(self):
		if not self.testing:
			self.user_actions = self.env.action_space.sample()
		else:
			self.user_actions = self.test_inputs[self.frame]
		self.frame += 1

	def after_step(self, RAM, input, screen):
		# visualiser.image = screen
		# visualiser.run()
		prediction = screen_reader.estimate_ram(screen)
		reference = torch.Tensor([list(self.env.data.memory.blocks[0])]).to(device)
		# datavis = prediction - reference
		# datavis = reference.type(torch.ByteTensor).view((32, 32))
		# im = torch.ones((32, 32, 3), dtype=torch.uint8)
		# for x in range(0, 32):
		# 	for y in range(0, 32):
		# 		im[x][y] = torch.Tensor([datavis[x][y], datavis[x][y], datavis[x][y]])
		# # im *= datavis

		global test_every, test_result

		# ram_show.imshow(im.numpy())
		# self.data_vis = SimpleImageViewer()
		if not self.testing:
			optimize(prediction, reference)
		if self.frame >= test_every:
			print(test_result)
			screen_reader.train(False)
			self.frame = 0
			self.testing = True
			test_result = 0
			self.done()
			pass
		elif self.frame < test_length:
			test_result += sum(abs(prediction[0] - reference[0]))
			screen_reader.train(False)
			self.testing = True
		else:
			screen_reader.train(True)
			self.testing = False

		super(MyEmulator, self).after_step(RAM, input, screen)

	def done(self):
		screen_reader.save()
		super(MyEmulator, self).done()

emul = MyEmulator(fps=0, render=False)
emul.run()
sys.exit(0)