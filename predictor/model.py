import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os

print('version = ', torch.__version__)
print("cuda friendly = ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToTensor()])

class Predictor(nn.Module):

	def __init__(self):
		super(Predictor, self).__init__()
		self.head1 = nn.linear(167, 1024)
		self.head2 = nn.linear(1024, 2048)
		self.head3 = nn.linear(2048, 1024)
		self.head4 = nn.linear(1024, 79)



	def forward(self, input, hidden, user_action):
		x = torch.cat(input, hidden, user_action)
		return x

	# def prepare_input(self, RAM):
	# 	RAM = transform(RAM)
	# 	return RAM.unsqueeze(0).to(device)

	def estimate_ram(self, RAM):
		# input = self.prepare_input(RAM)
		out = self(input)
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