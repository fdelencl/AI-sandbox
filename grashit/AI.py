#!/usr/local/bin/python3

import retro
import random
from net import Net, device
from memory import Memory
import time
import torch.optim as optim
import torch
import torch.nn as nn

import signal
import sys


import torchvision.transforms as T

transform = T.Compose([T.ToTensor()])

env = retro.make(game='Gradius-Nes', state='Level1')
# env = retro.make(game='GradiusIII-Snes', state='Level1.Mode1.Shield')

# [??, B, Select, Start, up, down, left, right, A]

n = Net().to(device)
criterion = nn.L1Loss()

m = Memory(10000)
o = optim.RMSprop(n.parameters(), lr=0.5, momentum=1)


# m.load('loli.kitty')

stop = False
def signal_handler(sig, frame):
	global stop
	stop = True
signal.signal(signal.SIGINT, signal_handler)

done = False

user_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0]

def on_key_press(symbol, modifiers):
	if symbol == 65363:
		user_actions[2] = 1
	if symbol == 65361:
		user_actions[3] = 1
	if symbol == 65505:
		user_actions[1] = 1
		user_actions[0] = 1
	if symbol == 119:
		user_actions[4] = 1
	if symbol == 115:
		user_actions[5] = 1
	if symbol == 97:
		user_actions[6] = 1
	if symbol == 100:
		user_actions[7] = 1
	if symbol == 32:
		user_actions[8] = 1
	pass

def on_key_release(symbol, modifiers):
	if symbol == 65363:
		user_actions[2] = 0
	if symbol == 65361:
		user_actions[3] = 0
	if symbol == 65505:
		user_actions[1] = 0
		user_actions[0] = 0
	if symbol == 119:
		user_actions[4] = 0
	if symbol == 115:
		user_actions[5] = 0
	if symbol == 97:
		user_actions[6] = 0
	if symbol == 100:
		user_actions[7] = 0
	if symbol == 32:
		user_actions[8] = 0
	if symbol == 65307:
		signal_handler()
	pass

current_screen = env.reset()
last_screen = current_screen
state = current_screen - last_screen
env.render()

env.unwrapped.viewer.window.on_key_press = on_key_press
env.unwrapped.viewer.window.on_key_release = on_key_release

def optimize():
	if len(m) < 1000:
		return 
	batch = m.sample(1000)
	actions = torch.zeros((1000, 9, 2), dtype=torch.float).to(device)
	states = torch.zeros((1000, 3, 224, 240), dtype=torch.float).to(device)
	rewards = torch.zeros((1000), dtype=torch.float).to(device)
	for i in range(0, 1000):
		states[i] = n.prepare_input(batch[i][0])
		# actions[i] = batch[i][1]
		rewards[i] = batch[i][3] + 1
		for j in range(0, 9):
			actions[i][j][batch[i][1][j]] = rewards[i]

	policy = n(states)
	n.zero_grad()
	loss = criterion(policy, actions)
	loss.backward()
	o.step()


def loop():
	global current_screen, last_screen, state, next_state
	last_screen = current_screen
	action = n.select_action(state)
	current_screen, _rew, done, _info = env.step(action)

	if not done and _info['lives'] == 3:
		next_state = current_screen - last_screen
		if not (_rew == 0 and random.randint(0, 3) <= 2):
			m.push(state, action, None, 0)
	else:
		for i in range(0, 3):
			optimize()
		env.reset()
		m.push(state, action, None, -10000)
	state = next_state
	env.render()

fps = 50
skipticks = 0
while not stop:
	tim1 = time.perf_counter()
	loop()
	tim2 = time.perf_counter()
	if tim2 > tim1 + skipticks:
		pass
	else:
		time.sleep(tim1 + skipticks - tim2)




m.save('loli')
env.close()
sys.exit(0)