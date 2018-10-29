import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random

print("cuda friendly = ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# [??, B, Select, Start, up, down, left, right, A]

transform = T.Compose([T.ToTensor()])

def prob(a):
	c = random.uniform(0, a[0] + a[1])
	if (c < a[0]):
		return 0
	else:
		return 1

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=10, stride=4)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=10, stride=4)
		self.bn4 = nn.BatchNorm2d(32)
		self.head1 = nn.Linear(4992, 512)
		self.lstm = nn.LSTMCell(512, 512)
		self.head2 = nn.Linear(512, 18)

		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)

	def forward(self, inputs):
		input, (hx, cx) = inputs
		x = self.conv1(input)
		x = F.relu(self.bn1(x))
		x = self.conv4(x)
		x = F.relu(self.bn4(x))
		x = F.relu(self.head1(x.view(x.size(0), -1)))
		hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
		x = hx
		x = F.relu(self.head2(x))
		
		return x.view((x.size(0), 9, 2)), (hx, cx)

	def prepare_input(self, screen):
		# inp = []
		# for key, value in info.items():
		#     inp.append(value)
		# print(inp)
		# print(len(inp))
		inp = transform(screen)
		# inp = torch.Tensor(inp)
		# print(inp)
		return inp.unsqueeze(0).to(device)

	# def prepare_input(self, screen):
	# 	screen = transform(screen)
	# 	return screen.unsqueeze(0).to(device)

	def select_action(self, info, mem):

		input = self.prepare_input(info)
		out, mem = self((input, mem))
		# with probabilities
		actions = [prob(a) for a in out[0]]
		#without probabilities
		# actions = out[0].max(1)[1]
		return actions, out, mem
