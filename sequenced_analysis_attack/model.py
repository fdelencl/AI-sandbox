import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os

print('version = ', torch.__version__)
print("cuda friendly = ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToTensor()])

class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()

		self.lstm = nn.LSTMCell(2048, 1024)

		self.actor_linear_1 = nn.Linear(1024, 1024)
		self.actor_linear_2 = nn.Linear(1024, 1024)
		self.actor_linear_3 = nn.Linear(1024, 1024)
		self.actor_linear_4 = nn.Linear(1024, 1024)
		self.actor_linear_5 = nn.Linear(1024, 18)

		self.critic_linear = nn.Linear(1024, 1)

	def forward(self, input, hidden):
		(hx, cx) = hidden

		hx, cx = self.lstm(x, (hx, cx))

		x =  F.relu(self.actor_linear_1(hx))
		x =  F.relu(self.actor_linear_2(x))
		x =  F.relu(self.actor_linear_3(x))
		x =  F.relu(self.actor_linear_4(x))
		x =  F.relu(self.actor_linear_5(x))
		act = ene.view(x.size(0), 9, 2)

		val = self.critic_linear(hx)
		return (act, val), (hx, cx)

	def prepare_input(self, screen):
		screen = transform(screen)
		return screen.unsqueeze(0).to(device)

	def estimate(self, screen, hidden):
		input = self.prepare_input(screen)
		out, hidden = self(input, hidden)

		return out, hidden

	def load(self, path='./screen_reader.model'):
		if os.path.isfile(path):
			self.load_state_dict(torch.load(path))
		else:
			print('cannot load screen reader from path: ', path)
		return

	def save(self, path='./screen_reader.model'):
		torch.save(self.state_dict(), path)
		return