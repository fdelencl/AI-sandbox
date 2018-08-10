import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

print("cuda friendly = ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# [??, B, Select, Start, up, down, left, right, A]

transform = T.Compose([T.ToTensor()])

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn4 = nn.BatchNorm2d(32)
		self.head = nn.Linear(4224, 18)

	def forward(self, input):
		x = self.conv1(input)
		x = F.relu(self.bn1(x))
		x = self.conv2(x)
		x = F.relu(self.bn2(x))
		x = self.conv3(x)
		x = F.relu(self.bn3(x))
		x = self.conv4(x)
		x = F.relu(self.bn4(x))
		x = self.head(x.view(x.size(0), -1))
		return x.view((x.size(0), 9, 2))

	def prepare_input(self, screen):
		screen = transform(screen)
		return screen.unsqueeze(0).to(device)

	def select_action(self, screen):
		input = self.prepare_input(screen)
		out = self(input)
		out = out[0].max(1)[1]
		return out

