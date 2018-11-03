import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functionalimport torchvision.transform as T

print("cuda friendly = ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToTensor()])

class screen_reader(nn.Module):

	def __init__(self):
		super(screen_reader, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn4 = nn.BatchNorm2d(32)
		self.head1 = nn.Linear(4224, 2048)
		self.head2 = nn.Linear(2048, 1024)

	def forward(self, input):
		x = self.conv1(input)
		x = self.bn1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.conv4(x)
		x = self.bn4(x)
		x = self.head1(x)
		x = self.head2(x)
		return x

	def prepare_input(self, screen):
		screen = transform(screen)
		return screen.unsqueeze(0).to(device)

	def estimate_ram(self, screen):
		input = self.prepare_input(screen)
		out = self(input)
		return out