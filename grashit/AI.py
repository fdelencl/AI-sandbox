#!/usr/local/bin/python3

import retro
import random
from net import Net, device
from memory import Memory
import time
import torch.optim as optim
import torch
import torch.nn as nn
from statistics import mean
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt


import signal
import sys


import torchvision.transforms as T

transform = T.Compose([T.ToTensor()])

env = retro.make(game='Gradius-Nes', state='./Level1', scenario='./roms/scenario.json', info='./roms/data.json')
# env = retro.make(game='GradiusIII-Snes', state='Level1.Mode1.Shield')

# [??, B, Select, Start, up, down, left, right, A]

n = Net().to(device)
criterion = nn.L1Loss()
# criterion = nn.CrossEntropyLoss()

# m = Memory(140000)
o = optim.RMSprop(n.parameters(), lr=0.0005, momentum=0.0001)


# m.load('loli.kitty')

stop = False
def signal_handler():
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

# screens = [env.reset(), 0, 0, 0, 0]
state = env.reset()
current_screen = 0
env.render()
state, _rew, done, _info = env.step(user_actions)

env.unwrapped.viewer.window.on_key_press = on_key_press
env.unwrapped.viewer.window.on_key_release = on_key_release

frames = []
average_f = []
scores = []
average_s = []
bonuses = []
average_b = []
profile = []
factors = []

acumulated_reward = 0
frame = 0

cx = torch.zeros(1, 512).to(device)
hx = torch.zeros(1, 512).to(device)
bonus_factor = 0
actions = []
outs = []

best_lifetime = 0;
best_score = 0

plt.ion()
plt.show()

def plotResults():
	global frames, scores, average_s, average_f, bonuses, average_b, profile, factors
	plt.figure(1)
	plt.clf()
	plt.title('scores')
	plt.plot(scores, 'b')
	plt.plot(average_s, 'g')
	plt.figure(2)
	plt.clf()
	plt.title('lifespan')
	plt.plot(frames, 'r')
	plt.plot(average_f, 'g')
	plt.figure(3)
	plt.clf()
	plt.title('bonus')
	plt.plot(bonuses, 'y')
	plt.plot(average_b, 'g')
	plt.figure(4)
	plt.clf()
	plt.title('profile')
	plt.plot(profile, 'r')
	plt.figure(5)
	plt.clf()
	plt.title('factor')
	plt.plot(factors, 'c')
	plt.hlines(1, 0, len(factors), 'g')
	plt.pause(0.001)

def optimize(outputs, frame, score, bonus_factor):
	global frames, scores, average_f, average_s, profile, best_lifetime, best_score, factors, actions
	blowit = 1
	if frame > best_lifetime:
		best_lifetime = frame
		blowit = 40
	
	if score > best_score:
		best_score = score
		blowit = 15

	print('best_lifetime = ', best_lifetime)
	print('best_score = ', best_score)
	print('frame = ', frame)
	print('score = ', score)
	frames.append(frame)

	score /= frame
	bonus_factor /= frame

	scores.append(score)
	bonuses.append(bonus_factor)

	average_frames = mean(frames)
	average_f.append(average_frames)

	average_scores = mean(scores)
	average_s.append(average_scores)

	average_bonus = mean(bonuses)
	average_b.append(average_bonus)

	if average_scores != 0:
		factor = frame / average_frames * 0.53 + score / average_scores * 0.25 + bonus_factor/average_bonus * 0.20
	else:
		factor = (frame / average_frames * 0.53) + bonus_factor/average_bonus * 0.20
	factors.append(factor)

	expectations = torch.cat(outputs, dim=0)
	profile = []
	for i in range(0, len(outputs)):
		profile.append(sum([1 / abs((i-f)) * (1 - i / len(outputs)) if i != f else 1 for f in frames]))

	for i in range(0, len(outputs)):
		expectations[i] = outputs[i]
		expectations[i] *= (profile[i] * frame / len(frames)  + 1)
		expectations[i][actions[i]] *= factor * blowit
		expectations[i][[((p + 1) % 2) for p in actions[i]]] /= factor * blowit



	# for i in range(0, len(outputs)):
	# 	expectations[i] = expectations[i] * (profile[i] + 0.5)
	expectations = expectations.detach()
	# print(outputs)
	# print(expectations)

	loss = criterion(torch.cat(outputs, dim=0), expectations)
	print('loss = ', loss)
	loss.backward()
	o.step()
	plotResults()
	print('--------------------------------------------\n\n')

def loop():
	# global current_screen, screens, acumulated_reward, frame
	global best_lifetime, best_score, _info, actions, outs, acumulated_reward, frame, frames, cx, hx, bonus_factor, state
	action, out, (hx, cx) = n.select_action(state, (cx, hx))
	actions.append(action)
	outs.append(out)
	state, _rew, done, _info = env.step(action)

	acumulated_reward += _rew
	frame += 1

	if not done and _info['lives'] == 3:
		bonus_factor = 0.3*_info['bonus.double|laser'] + 0.4*_info['bonus.missile'] + 1 * _info['bonus.options'] + 0.2 * _info['bonus.speed'] + 0.1
		pass
	else:
		optimize(outs, frame, acumulated_reward, bonus_factor)
		frame = 0
		outs = []
		actions = []
		acumulated_reward = 0
		cx = torch.zeros(1, 512).to(device)
		hx = torch.zeros(1, 512).to(device)
		state = env.reset()
		n.zero_grad()
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


env.close()
sys.exit(0)