import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os

print('version = ', torch.__version__)
print("cuda friendly = ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToTensor()])

class ScreenReader(nn.Module):

	def __init__(self):
		super(ScreenReader, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn4 = nn.BatchNorm2d(32)
		self.head0 = nn.Linear(21600, 5000)
		self.lstm = nn.LSTMCell(5000, 2048)
		self.head1 = nn.Linear(2048, 1024)
		self.head2 = nn.Linear(1024, 79)

	def forward(self, input, hidden):
		hx, cx = hidden
		x = F.relu(self.conv1(input))
		x = self.bn1(x)
		x = F.relu(self.conv2(x))
		x = self.bn2(x)
		x = F.relu(self.conv4(x))
		x = self.bn4(x)
		x = x.view(x.size(0), -1)
		x = F.relu(self.head0(x))
		hx, cx = self.lstm(x, (hx, cx))
		x = F.relu(self.head1(hx))
		x = F.relu(self.head2(x))
		return x, (hx, cx)

	def prepare_input(self, screen):
		screen = transform(screen)
		return screen.unsqueeze(0).to(device)

	def estimate_ram(self, screen, hidden):
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