from memory_pipe.recorder import Recorder
from emulator import Emulator
from screen_reader.model import ScreenReader, device
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

screen_reader = ScreenReader().to(device)
screen_reader.load()

learning_rate = 0.001
momentum = 0.00001
batch_size = 1000
hx = torch.zeros((1, 1536), dtype=torch.float, requires_grad=True).to(device)
cx = torch.zeros((1, 1536), dtype=torch.float, requires_grad=True).to(device)

criterion = nn.L1Loss()
optimizer = optim.RMSprop(screen_reader.parameters(), lr=learning_rate, momentum=momentum)

screens = torch.zeros((batch_size, 3, 224, 240), dtype=torch.float, requires_grad=True).to(device)
predicted = torch.zeros((batch_size, 1536), dtype=torch.float, requires_grad=True).to(device)
referenced = torch.zeros((batch_size, 1536), dtype=torch.float, requires_grad=True).to(device)

epoch = 0
def optimize():
	global epoch
	epoch += 1
	loss = criterion(predicted, referenced)
	print(epoch, loss.data)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	

class MyEmulator(Emulator):
	def __init__(self, fps=50, render=True, info=None, recorder=None):
		super(MyEmulator, self).__init__(fps=fps, render=render, info=info, recorder=recorder)
		self.batch = 0

	def before_step(self):
		self.user_actions = self.env.action_space.sample()


	def after_step(self, RAM, input, screen, info):
		global predicted, referenced, batch_size, screens, prediction, hx, cx
		prediction, (hx, cx) = screen_reader.estimate_ram(screen, (hx, cx))
		reference = torch.Tensor([list(RAM[0:256] + RAM[768:2048])]).to(device)

		screens[self.batch] = screen_reader.prepare_input(screen)[0]
		predicted[self.batch] = prediction[0]
		referenced[self.batch] = reference.to(torch.float)

		i = np.array(list(RAM[0:2048]))
		p = prediction.byte()[0]
		percieved_view = screen

		percieved_view[(i[800]) % 224][(i[864]+ 1) % 240] = [0, 255, 0]
		percieved_view[(i[800]) % 224][(i[864]+ 2) % 240] = [0, 255, 0]
		percieved_view[(i[800]) % 224][(i[864]+ 3) % 240] = [0, 255, 0]
		percieved_view[(i[800]) % 224][(i[864]- 1) % 240] = [0, 255, 0]
		percieved_view[(i[800]) % 224][(i[864]- 2) % 240] = [0, 255, 0]
		percieved_view[(i[800]) % 224][(i[864]- 3) % 240] = [0, 255, 0]

		percieved_view[(i[800]+ 1) % 224][(i[864]) % 240] = [0, 255, 0]
		percieved_view[(i[800]+ 2) % 224][(i[864]) % 240] = [0, 255, 0]
		percieved_view[(i[800]+ 3) % 224][(i[864]) % 240] = [0, 255, 0]
		percieved_view[(i[800]- 1) % 224][(i[864]) % 240] = [0, 255, 0]
		percieved_view[(i[800]- 2) % 224][(i[864]) % 240] = [0, 255, 0]
		percieved_view[(i[800]- 3) % 224][(i[864]) % 240] = [0, 255, 0]

		percieved_view[(p[288]- 1) % 224][(p[352]+ 1) % 240] = [255, 0, 0]
		percieved_view[(p[288]+ 1) % 224][(p[352]+ 1) % 240] = [255, 0, 0]
		percieved_view[(p[288]- 1) % 224][(p[352]- 1) % 240] = [255, 0, 0]
		percieved_view[(p[288]+ 1) % 224][(p[352]- 1) % 240] = [255, 0, 0]
		game.imshow(percieved_view)

		zoomed = np.zeros((3, 384, 192), dtype=np.uint8)
		zoomed[0][0:48] = np.kron(p[0:256].view(8, 32).detach().cpu().numpy(), np.ones((6, 6)))
		zoomed[0][144:384] = np.kron(p[256:1536].view(40, 32).detach().cpu().numpy(), np.ones((6, 6)))
		# zoomed[1] = np.kron(i.reshape((64, 32)), np.ones((6, 6)))
		# zoomed[2][48:144] = np.kron(i[256:768].reshape((16, 32)), np.ones((6, 6)))
		zoomed = np.transpose(zoomed, (1, 2, 0))
		ram_show.imshow(zoomed)

		if self.batch >= batch_size -1:
			optimize()
			hx = torch.zeros((1, 1536), dtype=torch.float, requires_grad=True).to(device)
			cx = torch.zeros((1, 1536), dtype=torch.float, requires_grad=True).to(device)
			screens = torch.zeros((batch_size, 3, 224, 240), dtype=torch.float, requires_grad=True).to(device)
			predicted = torch.zeros((batch_size, 1536), dtype=torch.float, requires_grad=True).to(device)
			referenced = torch.zeros((batch_size, 1536), dtype=torch.float, requires_grad=True).to(device)
			self.batch = 0
		else:
			self.batch += 1
		super(MyEmulator, self).after_step(RAM, input, screen, info)

	def done(self):
		screen_reader.save()
		super(MyEmulator, self).done()

class MyRecorder(Recorder):
	def record(self, RAM, input, screen):
		self.session['RAMs'].append(RAM)
		self.session['inputs'].append(input)

# rec = MyRecorder()
emul = MyEmulator(fps=0, render=False, info='./rom/data.json', recorder=None)
emul.run()
sys.exit(0)