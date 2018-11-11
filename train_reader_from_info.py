from memory_pipe.recorder import Recorder
from emulator import Emulator
from screen_reader.model_from_info import ScreenReader, device
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
model_path = './rom/screen_reader_from_info_WIP.model'
screen_reader.load(model_path)

learning_rate = 0.0005
momentum = 0.001
batch_size = 1000

criterion = nn.L1Loss()
optimizer = optim.RMSprop(screen_reader.parameters(), lr=learning_rate, momentum=momentum)

screens = torch.zeros((batch_size, 3, 224, 240), dtype=torch.float, requires_grad=True).to(device)
predicted = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
referenced = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)


def optimize():
	optimizer.zero_grad()
	predicted = screen_reader(screens)
	loss = criterion(predicted, referenced)
	print(loss)
	loss.backward()
	optimizer.step()

class MyEmulator(Emulator):
	def __init__(self, fps=50, render=True, info=None):
		super(MyEmulator, self).__init__(fps=fps, render=render, info=info)
		self.batch = 0

	def before_step(self):
		self.user_actions = self.env.action_space.sample()


	def after_step(self, RAM, input, screen, info):
		global predicted, referenced, batch_size, screens
		inp = []

		prediction = screen_reader.estimate_ram(screen)
		for key, value in info.items():
			inp.append(value)
		inp = torch.Tensor(inp)
		reference = inp

		screens[self.batch] = screen_reader.prepare_input(screen)[0]
		referenced[self.batch] = reference.to(torch.float)

		i = inp.byte()
		p = prediction.byte()[0]
		percieved_view = screen

		percieved_view[(i[78]) % 224][(i[72]+ 1) % 240] = [0, 255, 0]
		percieved_view[(i[78]) % 224][(i[72]+ 2) % 240] = [0, 255, 0]
		percieved_view[(i[78]) % 224][(i[72]+ 3) % 240] = [0, 255, 0]
		percieved_view[(i[78]) % 224][(i[72]- 1) % 240] = [0, 255, 0]
		percieved_view[(i[78]) % 224][(i[72]- 2) % 240] = [0, 255, 0]
		percieved_view[(i[78]) % 224][(i[72]- 3) % 240] = [0, 255, 0]

		percieved_view[(i[78]+ 1) % 224][(i[72]) % 240] = [0, 255, 0]
		percieved_view[(i[78]+ 2) % 224][(i[72]) % 240] = [0, 255, 0]
		percieved_view[(i[78]+ 3) % 224][(i[72]) % 240] = [0, 255, 0]
		percieved_view[(i[78]- 1) % 224][(i[72]) % 240] = [0, 255, 0]
		percieved_view[(i[78]- 2) % 224][(i[72]) % 240] = [0, 255, 0]
		percieved_view[(i[78]- 3) % 224][(i[72]) % 240] = [0, 255, 0]

		percieved_view[(p[78]- 1) % 224][(p[72]+ 1) % 240] = [255, 0, 0]
		percieved_view[(p[78]+ 1) % 224][(p[72]+ 1) % 240] = [255, 0, 0]
		percieved_view[(p[78]- 1) % 224][(p[72]- 1) % 240] = [255, 0, 0]
		percieved_view[(p[78]+ 1) % 224][(p[72]- 1) % 240] = [255, 0, 0]
		game.imshow(percieved_view)

		zoomed = np.zeros((30, 3, 790), dtype=np.uint8)
		
		zoomed[0:10] = np.kron(reference.cpu().numpy(), np.ones((10, 3, 10)))
		zoomed[10:20] = np.kron(p.detach().cpu().numpy(), np.ones((10, 3, 10))) 
		zoomed[20:30] = abs(zoomed[0:10] - zoomed[10:20])
		zoomed = np.transpose(zoomed, (0, 2, 1))
		ram_show.imshow(zoomed)

		# percieved_view = screen

		# percieved_view[(p[0][800]- 1) % 224][(p[0][864]+ 1) % 240] = [255, 0, 0]
		# percieved_view[(p[0][800]+ 1) % 224][(p[0][864]+ 1) % 240] = [255, 0, 0]
		# percieved_view[(p[0][800]- 1) % 224][(p[0][864]- 1) % 240] = [255, 0, 0]
		# percieved_view[(p[0][800]+ 1) % 224][(p[0][864]- 1) % 240] = [255, 0, 0]
		# game.imshow(percieved_view)
		# print('player position reference : ', prediction[0][864], prediction[0][800])
		# print('player position prediction: ', reference[0][864], reference[0][800])


		if self.batch >= batch_size -1:
			screen_reader.train()
			optimize()
			screen_reader.eval()
			screens = torch.zeros((batch_size, 3, 224, 240), dtype=torch.float, requires_grad=True).to(device)
			predicted = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
			referenced = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
			self.batch = 0
		else:
			self.batch += 1
		super(MyEmulator, self).after_step(RAM, input, screen, info)

	def done(self):
		global model_path
		screen_reader.save(model_path)
		super(MyEmulator, self).done()

emul = MyEmulator(fps=0, render=False, info='./rom/data.json')
emul.run()
sys.exit(0)