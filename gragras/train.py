#!/usr/local/bin/python3

import retro
import random
from net import Net, device
import time
import torch.optim as optim
import torch
import torch.nn as nn
from statistics import mean
from math import exp
import os.path
import matplotlib
import matplotlib.pyplot as plt
import pickle


import signal
import sys


import torchvision.transforms as T

transform = T.Compose([T.ToTensor()])

env = retro.make(game='Gradius-Nes', state='./Level1', scenario='./roms/scenario.json', info='./roms/data.json')
# env = retro.make(game='GradiusIII-Snes', state='Level1.Mode1.Shield')

# [??, B, Select, Start, up, down, left, right, A]
if os.path.isfile('./records'):
	with open('./records', "rb") as fp:
		save = pickle.load(fp)
		frames = save["frames"]
		average_f = save["average_f"]
		scores = save["scores"]
		average_s = save["average_s"]
		bonuses = save["bonuses"]
		average_b = save["average_b"]
		profile = save["profile"]
		factors = save["factors"]
else:
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

# n = Net().to(device)
# if os.path.isfile('./model'):
# 	n.load_state_dict(torch.load('./model'))

if os.path.isfile('./model'):
	n = torch.load('./model')
else:
	n = Net().to(device)


criterion = nn.L1Loss()
# criterion = nn.CrossEntropyLoss()

# m = Memory(140000)
o = optim.RMSprop(n.parameters(), lr=0.1, momentum=0.1)


# m.load('loli.kitty')

stop = False
def signal_handler():
	global stop
	torch.save(n, './model')
	# torch.save(nstate_dict(), './model')
	with open("records", "wb") as fp:
		save = {
			"frames": frames,
			"average_f": average_f,
			"scores": scores,
			"average_s": average_s,
			"bonuses": bonuses,
			"average_b": average_b,
			"profile": profile,
			"factors": factors
		}
		pickle.dump(save, fp)
	stop = True
signal.signal(signal.SIGINT, signal_handler)

done = False
master = False
user_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0]

def on_key_press(symbol, modifiers):
	global master, user_actions
	print(symbol)
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
	if symbol == 65293:
		master = True
	if symbol == 65288:
		master = False
	pass

def on_key_release(symbol, modifiers):
	global master, user_actions
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
# state, _rew, done, _info = env.step(user_actions)

env.unwrapped.viewer.window.on_key_press = on_key_press
env.unwrapped.viewer.window.on_key_release = on_key_release

cx = torch.zeros(1, 1024).to(device)
hx = torch.zeros(1, 1024).to(device)
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
	plt.title('factor')
	plt.plot(factors, 'c')
	plt.hlines(1, 0, len(factors), 'g')
	plt.figure(5)
	plt.clf()
	plt.title('profile')
	plt.plot(profile, 'r')
	plt.pause(0.001)

def optimize(outputs, frame, score, bonus_factor):
	global frames, scores, average_f, average_s, profile, best_lifetime, best_score, factors, actions
	global master, user_actions
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
	score /= frame
	bonus_factor /= frame
	frames.append(frame)
	scores.append(score)
	bonuses.append(bonus_factor)

	average_frames = mean(frames[-250:])
	average_f.append(average_frames)

	average_scores = mean(scores[-250:])
	average_s.append(average_scores)

	average_bonus = mean(bonuses[-250:])
	average_b.append(average_bonus)

	if average_scores != 0:
		factor = frame / average_frames * 0.53 + score / average_scores * 0.25 + bonus_factor/average_bonus * 0.20
	else:
		factor = (frame / average_frames * 0.53) + bonus_factor/average_bonus * 0.25
	factors.append(factor)

	print('factor = ', factor)
	
	expectations = torch.cat(outputs, dim=0)
	profile = []
	for i in range(0, len(outputs)):
		profile.append(sum([(1 - exp(-30 / ((i - f) ** 2))) * 0.5 if i != f and f < frame - 350 else 1 if f < frame - 350 else 0 for f in frames]) * exp(-100 / ((i - frame - 5) ** 2)))
	profile = [p / frame + 1 for p in profile]
	for i in range(0, len(outputs)):
		for j in range(0, len(actions[i])):
			if actions[i][j] == 0:
				expectations[i][j] = expectations[i][j] / (factor * blowit * profile[i])
			else:
				expectations[i][j] = expectations[i][j] * (factor * blowit * profile[i])
		# expectations[i] = actions[i] - expectations[i]) / (factor * blowit * profile[i])

		# for j in range(0, len(expectations[i])):
		# 	expectations[i][j][actions[i][j]] *= factor * blowit * profile[i]
		# 	expectations[i][j][(actions[i][j] + 1) % 2] /= factor * blowit * profile[i]
			# expectations[i][j][(actions[i][j] + 1) % 2] /= factor
		# expectations[i][actions[i]] *= factor * blowit * profile[i]
		# print(expectations[i])
		# print(actions[i])
		# for j in range(0, len(actions[i])):
		# 	expectations[i][actions[i]] *= factor * blowit * profile[i]




	# for i in range(0, len(outputs)):
	# 	expectations[i] = expectations[i] * (profile[i] + 0.5)
	expectations = expectations.detach()

	loss = criterion(torch.cat(outputs, dim=0), expectations)
	print('loss = ', loss)
	loss.backward()
	o.step()
	plotResults()
	print('--------------------------------------------\n\n')

def loop():
	# global current_screen, screens, acumulated_reward, frame
	global best_lifetime, best_score, _info, actions, outs, acumulated_reward, frame, frames, cx, hx, bonus_factor
	global master, user_actions
	action, out, (hx, cx) = n.select_action(env.data.memory.blocks[0], (cx, hx))
	actions.append(action)
	outs.append(out)
	if master:
		state, _rew, done, _info = env.step(user_actions);
	else:
		state, _rew, done, _info = env.step(action)
	# print([789], _info["enemy_1.type"])

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
		cx = torch.zeros(1, 1024).to(device)
		hx = torch.zeros(1, 1024).to(device)
		state = env.reset()
		n.zero_grad()
	env.render()

fps = 50
skipticks = 1 / fps
while not stop:
	tim1 = time.perf_counter()
	loop()
	tim2 = time.perf_counter()
	if not master or tim2 > tim1 + skipticks:
		pass
	else:
		time.sleep(tim1 + skipticks - tim2)


env.close()
sys.exit(0)