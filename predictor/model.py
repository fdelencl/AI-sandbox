import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import os

print('version = ', torch.__version__)
print("cuda friendly = ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToTensor()])

class Predictor(nn.Module):

	def __init__(self):
		super(Predictor, self).__init__()
		# self.head1 = nn.Linear(167, 1024)
		self.head1 = nn.Linear(88, 1024)
		self.head2 = nn.Linear(1024, 2048)
		self.head3 = nn.Linear(2048, 1024)
		self.head4 = nn.Linear(1024, 512)
		self.idh = nn.Linear(512, 79)
		self.head5 = nn.Linear(512, 1)

	def forward(self, input, actions):
		x = torch.cat((input, actions), 1)
		x = self.head1(x)
		x = self.head2(x)
		x = self.head3(x)
		x = self.head4(x)
		# hidden = self.idh(x)
		x = self.head5(x)
		# return F.relu(x), hidden.detach()
		return F.relu(x)


	def estimate_value(self, RAM, actions):
		actions = torch.Tensor(actions).to(device)
		# out, hidden = self(RAM, hidden, actions)
		out = self(RAM, actions)
		return out

	def load(self, path='./predictor.model'):
		if os.path.isfile(path):
			self.load_state_dict(torch.load(path))
		else:
			print('cannot load predictor from path: ', path)
		return

	def save(self, path='./predictor.model'):
		torch.save(self.state_dict(), path)
		return