from memory_pipe.recorder import Recorder
from memory_pipe.reader import Reader
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
screen_reader.load('./rom/screen_reader_with_info.model')

learning_rate = 0.0001
momentum = 0.00001
batch_size = 500
hx = torch.zeros((1, 2048), dtype=torch.float, requires_grad=True).to(device)
cx = torch.zeros((1, 2048), dtype=torch.float, requires_grad=True).to(device)

criterion = nn.L1Loss()
optimizer = optim.RMSprop(screen_reader.parameters(), lr=learning_rate, momentum=momentum)

screens = torch.zeros((batch_size, 3, 224, 240), dtype=torch.float, requires_grad=True).to(device)
predicted = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
referenced = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)

epoch = 0
def optimize():
	global epoch
	epoch += 1
	loss = criterion(predicted, referenced)
	print(epoch, loss.data)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	screen_reader.zero_grad()
	
	

class MyEmulator(Emulator):
	def __init__(self, fps=50, render=True, info=None, recorder=None, reader=None):
		super(MyEmulator, self).__init__(fps=fps, render=render, info=info, recorder=recorder, reader=reader)
		self.batch = 0

	def before_step(self):
		self.user_actions = self.env.action_space.sample()


	def after_step(self, RAM, input, screen, info):
		global predicted, referenced, batch_size, screens, prediction, hx, cx
		prediction, (hx, cx) = screen_reader.estimate_ram(screen, (hx, cx))
		inp = []
		index = 0
		for key, value in info.items():
			if key == 'enemy_0.type':
				print('enemy_0.type', )
			inp.append(value)
		reference = torch.Tensor([inp])

		p = prediction[0]
		point_orders = [
			[1 , 2 ],
			[58, 24],
			[55, 47],
			[53, 17],
			[40, 6 ],
			[46, 45],
			[3 , 41],
			[23, 30],
			[61, 32],
			[25, 26],
			[68, 22]
		]

		point_indexes = [
				[[1 , 2 ], [ p[1 ], p[2 ] ], [ inp[1 ], inp[2 ] ]], #0
				[[58, 24], [ p[58], p[24] ], [ inp[58], inp[24] ]], #1
				[[55, 47], [ p[55], p[47] ], [ inp[55], inp[47] ]], #2
				[[53, 17], [ p[53], p[17] ], [ inp[53], inp[17] ]], #3
				[[40, 6 ], [ p[40], p[6 ] ], [ inp[40], inp[6 ] ]], #4
				[[46, 45], [ p[46], p[45] ], [ inp[46], inp[45] ]], #5
				[[3 , 41], [ p[3 ], p[41] ], [ inp[3 ], inp[41] ]], #6
				[[23, 30], [ p[23], p[30] ], [ inp[23], inp[30] ]], #7
				[[61, 32], [ p[61], p[32] ], [ inp[61], inp[32] ]], #8
				[[25, 26], [ p[25], p[26] ], [ inp[25], inp[26] ]], #9
				[[68, 22], [ p[68], p[22] ], [ inp[68], inp[22] ]]  #a
			]

		sorted_points = sorted(point_indexes, reverse=True , key=lambda pt: pt[2][0] * 255 + pt[2][1])

		n = 0
		for pt, [x, y], [rx, ry] in sorted_points:
			p[point_orders[n][0]] = x
			p[point_orders[n][1]] = y
			reference[0][point_orders[n][0]] = rx
			reference[0][point_orders[n][1]] = ry
			n += 1


		# reference = torch.Tensor([list(RAM[0:256] + RAM[768:2048])]).to(device)

		screens[self.batch] = screen_reader.prepare_input(screen)[0]
		predicted[self.batch] = p
		referenced[self.batch] = reference.to(torch.float)

		# i = np.array(list(RAM[0:2048]))
		p = p.type(torch.IntTensor)
		percieved_view = screen

		variables_to_display = [
			[72, 78, [255, 0, 0]],
			[1 , 2 , [255, 255, 255]],
			[58, 24, [255, 255, 255]],
			[55, 47, [255, 255, 255]],
			[53, 17, [255, 255, 255]],
			[40, 6 , [255, 255, 255]],
			[46, 45, [255, 255, 255]],
			[3 , 41, [255, 255, 255]],
			[23, 30, [255, 255, 255]],
			[61, 32, [255, 255, 255]],
			[25, 26, [255, 255, 255]],
			[68, 22, [255, 255, 255]]
		]
		for [x, y, color] in variables_to_display:
			percieved_view[(p[y]) % 224][(p[x]+ 1) % 240] = color
			percieved_view[(p[y]) % 224][(p[x]+ 2) % 240] = color
			percieved_view[(p[y]) % 224][(p[x]+ 3) % 240] = color
			percieved_view[(p[y]) % 224][(p[x]- 1) % 240] = color
			percieved_view[(p[y]) % 224][(p[x]- 2) % 240] = color
			percieved_view[(p[y]) % 224][(p[x]- 3) % 240] = color

			percieved_view[(p[y]+ 1) % 224][(p[x]) % 240] = color
			percieved_view[(p[y]+ 2) % 224][(p[x]) % 240] = color
			percieved_view[(p[y]+ 3) % 224][(p[x]) % 240] = color
			percieved_view[(p[y]- 1) % 224][(p[x]) % 240] = color
			percieved_view[(p[y]- 2) % 224][(p[x]) % 240] = color
			percieved_view[(p[y]- 3) % 224][(p[x]) % 240] = color

		game.imshow(percieved_view)

		# zoomed = np.zeros((3, 384, 192), dtype=np.uint8)
		# zoomed[0][0:48] = np.kron(p[0:256].view(8, 32).detach().cpu().numpy(), np.ones((6, 6)))
		# zoomed[0][144:384] = np.kron(p[256:1536].view(40, 32).detach().cpu().numpy(), np.ones((6, 6)))
		# # zoomed[1] = np.kron(i.reshape((64, 32)), np.ones((6, 6)))
		# # zoomed[2][48:144] = np.kron(i[256:768].reshape((16, 32)), np.ones((6, 6)))
		# zoomed = np.transpose(zoomed, (1, 2, 0))
		# ram_show.imshow(zoomed)

		if self.batch >= batch_size -1:
			optimize()
			hx = torch.zeros((1, 2048), dtype=torch.float, requires_grad=True).to(device)
			cx = torch.zeros((1, 2048), dtype=torch.float, requires_grad=True).to(device)
			screens = torch.zeros((batch_size, 3, 224, 240), dtype=torch.float, requires_grad=True).to(device)
			predicted = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
			referenced = torch.zeros((batch_size, 79), dtype=torch.float, requires_grad=True).to(device)
			self.batch = 0
		else:
			self.batch += 1
		super(MyEmulator, self).after_step(RAM, input, screen, info)

	def done(self):
		screen_reader.save('./rom/screen_reader_with_info.model')
		super(MyEmulator, self).done()

rec = Recorder()
# reader = Reader()
emul = MyEmulator(fps=0, render=False, info='./rom/data.json', recorder=None, reader=None)
emul.run()
sys.exit(0)