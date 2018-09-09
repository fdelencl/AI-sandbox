import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from utils import norm_col_init, weights_init, weights_init_mlp
import torch.nn.init as init
from torch.autograd import Variable

print('cuda friendly = ', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([T.ToTensor()])

class Net(nn.Module):
	def __init__(self, num_inputs, action_space):
		super(Net, self).__init__()
		self.conv1 = nn.Conv1d(num_inputs, 32, 3, stride=1, padding=1)
		self.lrelu1 = nn.LeakyReLU(0.1)
		self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
		self.lrelu2 = nn.LeakyReLU(0.1)
		self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
		self.lrelu3 = nn.LeakyReLU(0.1)
		self.conv4 = nn.Conv1d(64, 64, 1, stride=1)
		self.lrelu4 = nn.LeakyReLU(0.1)

		self.lstm = nn.LSTMCell(1600, 128)
		num_outputs = action_space.shape[0]
		self.critic_linear = nn.Linear(128, 1)
		self.actor_linear = nn.Linear(128, num_outputs)
		self.actor_linear2 = nn.Linear(128, num_outputs)

		self.apply(weights_init)
		lrelu_gain = nn.init.calculate_gain('leaky_relu')
		self.conv1.weight.data.mul_(lrelu_gain)
		self.conv2.weight.data.mul_(lrelu_gain)
		self.conv3.weight.data.mul_(lrelu_gain)
		self.conv4.weight.data.mul_(lrelu_gain)

		self.actor_linear.weight.data = norm_col_init(
			self.actor_linear.weight.data, 0.01)
		self.actor_linear.bias.data.fill_(0)
		self.actor_linear2.weight.data = norm_col_init(
			self.actor_linear2.weight.data, 0.01)
		self.actor_linear2.bias.data.fill_(0)
		self.critic_linear.weight.data = norm_col_init(
			self.critic_linear.weight.data, 1.0)
		self.critic_linear.bias.data.fill_(0)

		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)

		self.train()

	def forward(self, inputs):
		x, (hx, cx) = inputs

		x = self.lrelu1(self.conv1(x))
		x = self.lrelu2(self.conv2(x))
		x = self.lrelu3(self.conv3(x))
		x = self.lrelu4(self.conv4(x))

		x = x.view(x.size(0), -1)
		hx, cx = self.lstm(x, (hx, cx))
		x = hx

		return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)


	def prepare_input(self, screen):
		screen = transform(screen)
		return screen.unsqueeze(0).to(device)

	def select_action(self, screen, cells):
		input = self.prepare_input(screen)
		out = self((input, cells))
		print(out)
		out = out[0].max(1)[1]
		return out

