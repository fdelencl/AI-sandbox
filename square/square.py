import math as mat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.0015
momentum = 0.1

batch_size = 225
square_size = 100

square = Image.open('squares.png')

def myTransform(ten):
	for x in range(len(ten)):
		for y in range(len(ten[x])):
			for z in range(len(ten[x, y])):
				ten[x, y, z] = 1 if (ten[x, y, z] > 0) else 0
	return ten[0:1, :, :].view((y + 1, z + 1))

transform = T.Compose([
	T.ToTensor(),
	myTransform
	])
square = transform(square)
print(square)
print(square.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inputs_reference():
	inputs = torch.zeros((batch_size, 2), device=device, dtype=torch.float)
	reference = torch.zeros((batch_size), device=device, dtype=torch.long)
	r = int(mat.sqrt(batch_size))
	for x in range(r):
		for y in range(r):
			inputs[x * r + y, 0] = x * square_size / r
			inputs[x * r + y, 1] = y * square_size / r
			reference[x * r + y] = square[x * square_size / r, y * square_size / r]
	return inputs, reference

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(2, 200)
		self.ht1 = nn.Hardtanh(-2, 2)
		self.fc2 = nn.Linear(200, 1000)
		self.fc3 = nn.Linear(1000, 200)
		self.fc4 = nn.Linear(200, 2)
		self.sigm = nn.Sigmoid()

	def forward(self, x):
		x = self.ht1(self.fc1(x))
		x = F.relu(self.ht1(self.fc2(x)))
		x = self.ht1(self.fc3(x))
		x = self.fc4(x)
		x = self.sigm(x)
		return x

def print_model(generation, err):
	img = np.empty((square_size, square_size, 3), dtype=np.uint8)
	input = torch.zeros((square_size * square_size, 2), device=device, dtype=torch.float)
	for x in range(square_size):
		for y in range(square_size):
			input[x * square_size + y, 0] = x
			input[x * square_size + y, 1] = y
	output = net(input)
	for x in range(square_size):
		for y in range(square_size):
			img[x, y, 0] = int(255 * output[x * square_size + y,0])
			img[x, y, 1] = int(255 * output[x * square_size + y,1])
			img[x, y, 2] = 0
	image = Image.fromarray(img, mode='RGB')
	image.save('img' + str(generation) + '_' + str(err) + '.png')

def calc_err(prediction, reference):
	err = 0
	for i in range(len(reference)):
		if prediction[i, 0] >= prediction[i, 1] and reference[i] == 1:
			err += 1
		elif prediction[i, 1] >= prediction[i, 0] and reference[i] == 0:
			err += 1
	return err


net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

inputs, reference = inputs_reference()
agg = np.array([])
errors = np.array([])
sm = 2500
err = 2500
t = 0
while err > 0 and t < 500000:
	t += 1
	prediction = net(inputs)
	net.zero_grad()
	loss = criterion(prediction, reference)

	loss.backward()
	# print(loss)
	optimizer.step()
	err = calc_err(prediction, reference)
	errors = np.append(errors, err)
	if err < sm:
		if err < sm:
			sm = err
		print_model(t, err)
	print('error: ' + str(err) + ' generation: ' + str(t))
	# agg = np.append(agg, loss.item())


# print_model(t, loss.item())

fig, ax1 = plt.subplots()

ax1.set_xlabel('generation')
ax1.set_ylabel('errors')
ax1.plot(range(len(errors)), errors, color='tab:red')

# ax2 = ax1.twinx()
# ax2.set_ylabel('loss', color='tab:blue')
# ax2.plot(range(len(agg)), agg, color='tab:blue')
# ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.tight_layout()

plt.show()
