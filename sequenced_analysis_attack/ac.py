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
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=7, stride=2)
		self.bn4 = nn.BatchNorm2d(32)

		self.lstm = nn.LSTMCell(2880, 1024)

		self.enemy_linear0 = nn.Linear(2880, 45)
		self.enemy_linear1 = nn.Linear(45, 45)
		self.enemy_linear2 = nn.Linear(45, 45)
		self.enemy_linear3 = nn.Linear(45, 45)
		self.enemy_linear4 = nn.Linear(45, 45)
		self.enemy_linear5 = nn.Linear(45, 45)
		self.enemy_linear6 = nn.Linear(45, 45)
		self.enemy_linear7 = nn.Linear(45, 45)
		self.enemy_linear8 = nn.Linear(45, 45)
		self.enemy_linear9 = nn.Linear(45, 33)

		self.position_linear = nn.Linear(1024, 2)

		self.actor_linear = nn.Linear(1024, 18)

		self.critic_linear = nn.Linear(1024, 1)




	def forward(self, input, hidden):
		(hx, cx) = hidden
		x = F.relu(self.conv1(input))
		x = self.bn1(x)
		x = F.relu(self.conv2(x))
		x = self.bn2(x)
		x = F.relu(self.conv3(x))
		x = self.bn3(x)
		x = F.relu(self.conv4(x))
		x = self.bn4(x)
		x = x.view(x.size(0), -1)

		ene = self.enemy_linear0(x)
		ene = self.enemy_linear1(ene)
		ene = self.enemy_linear2(ene)
		ene = self.enemy_linear3(ene)
		ene = self.enemy_linear4(ene)
		ene = self.enemy_linear5(ene)
		ene = self.enemy_linear6(ene)
		ene = self.enemy_linear7(ene)
		ene = self.enemy_linear8(ene)
		ene = self.enemy_linear9(ene)
		ene = ene.view(ene.size(0), 11, 3)

		hx, cx = self.lstm(x, (hx, cx))
		pos = self.position_linear(hx)

		act = self.actor_linear(hx)
		act = act.view(act.size(0), 9, 2)
		act = F.softmax(act, dim=2)

		val = self.critic_linear(hx)
		return (pos, ene, act, val), (hx, cx)

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