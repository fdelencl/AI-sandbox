from emulator import Emulator
from sequenced_analysis_attack.model import Model, device

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import numpy as np
from math import log
from gym.envs.classic_control.rendering import SimpleImageViewer


game = SimpleImageViewer()

SAA = Model().to(device)
model_path = './rom/saa.model'
SAA.load(model_path)

learning_rate = 0.0001
momentum = 0.0001
batch_size = 100
hx = torch.zeros((1, 1024), dtype=torch.float, requires_grad=True).to(device)
cx = torch.zeros((1, 1024), dtype=torch.float, requires_grad=True).to(device)

gamma = 0.9
tau = 1.0
learning_rate = 0.0001
enemy_pos_criterion = nn.L1Loss()
player_pos_criterion = nn.L1Loss()
optimizer = optim.RMSprop(SAA.parameters(), lr=learning_rate, momentum=momentum)


class SAAEmulator(Emulator):
	def __init__(self, fps=0, info='./rom/data.json'):
		super(SAAEmulator, self).__init__(render=False, fps=fps, info=info)
		self.epoch = 0

	def done(self):
		global model_path
		SAA.save(model_path)
		super(SAAEmulator, self).done()

	def before_run(self):
		global current_score, batch_size, hx, cx, current_life
		current_score = 0
		current_life = 3
		hx = torch.zeros((1, 1024), dtype=torch.float, requires_grad=True).to(device)
		cx = torch.zeros((1, 1024), dtype=torch.float, requires_grad=True).to(device)
		self.reset()

	def reset(self):
		self.batch = 0
		self.rewards = []
		self.ref_positions = torch.zeros((batch_size + 1, 2), requires_grad=True).to(device)
		self.ref_enemies = torch.zeros((batch_size + 1, 11, 3), requires_grad=True).to(device)
		self.enemies = torch.zeros((batch_size + 1, 11, 3), requires_grad=True).to(device)
		self.positions = torch.zeros((batch_size + 1, 2), requires_grad=True).to(device)
		self.actions = []
		self.entropies = []
		self.values = []
		self.log_probs = []

	def actor_critic_loss(self):
		global gamma, tau, batch_size
		value_loss = 0
		policy_loss = 0
		R = torch.zeros(1, 1).to(device)
		R = self.values[batch_size].data
		gae = torch.zeros(1, 1).to(device)
		for i in reversed(range(batch_size - 1)):
			R = gamma * R + self.rewards[i]
			advantage = R - self.values[i]
			value_loss = value_loss + 0.5 * advantage.pow(2)

			delta_t = self.rewards[i] + gamma * self.values[i + 1] - self.values[i]
			gae = gae * gamma * tau + delta_t
			policy_loss = policy_loss - self.log_probs[i].sum() * gae - 0.01 * self.entropies[i]
		return value_loss, policy_loss

	def optimize(self, retain=False):
		global batch_size, hx, cx
		self.epoch += 1
		enemy_pos_loss = enemy_pos_criterion(self.enemies, self.ref_enemies)
		player_pos_loss = player_pos_criterion(self.positions, self.ref_positions)
		# value_loss, policy_loss = self.actor_critic_loss()
		loss = enemy_pos_loss + player_pos_loss
		loss.backward()
		optimizer.step()
		self.reset()
		optimizer.zero_grad()
		SAA.zero_grad()
		hx = hx.detach()
		cx = cx.detach()

	def after_step(self, RAM, input, screen, info):
		global current_score, current_life, batch_size, hx, cx
		# (position, enemies, prob, value), (hx, cx) = SAA.estimate(screen, (hx, cx))
		(position, enemies), (hx, cx) = SAA.estimate(screen, (hx, cx))

		# delta_score = info['score'] - current_score
		# if current_life != info['lives']:
		# 	reward = 0
		# else:
		# 	reward = 1
		# delta_score = info['score'] - current_score
		# reward += delta_score
		# self.rewards.append(reward)
		# current_score = info['score']
		# current_life = info['lives']

		self.ref_positions[self.batch] = torch.tensor([info['p1.x'], info['p1.y']])
		self.positions[self.batch] = position[0]

		enemies_positions = [
			[info['enemy_0.x'], info['enemy_0.y'], info['enemy_0.type']],
			[info['enemy_1.x'], info['enemy_1.y'], info['enemy_1.type']],
			[info['enemy_2.x'], info['enemy_2.y'], info['enemy_2.type']],
			[info['enemy_3.x'], info['enemy_3.y'], info['enemy_3.type']],
			[info['enemy_4.x'], info['enemy_4.y'], info['enemy_4.type']],
			[info['enemy_5.x'], info['enemy_5.y'], info['enemy_5.type']],
			[info['enemy_6.x'], info['enemy_6.y'], info['enemy_6.type']],
			[info['enemy_7.x'], info['enemy_7.y'], info['enemy_7.type']],
			[info['enemy_8.x'], info['enemy_8.y'], info['enemy_8.type']],
			[info['enemy_9.x'], info['enemy_9.y'], info['enemy_9.type']],
			[info['enemy_a.x'], info['enemy_a.y'], info['enemy_a.type']]
		]
		enemies_positions = list(map(lambda pt: [0, 0, 0] if pt[2] == 0 else pt, enemies_positions))
		sorted_enemies = sorted(enemies_positions, reverse=False , key=lambda pt: pt[1] * 255 + pt[0])
		self.ref_enemies[self.batch] = torch.tensor(sorted_enemies)


		p = enemies[0].type(torch.IntTensor).detach()
		percieved_view = screen

		for [x, y, _] in p:
			percieved_view[(y) % 224][(x+ 1) % 240] = [0, 0, 255]
			percieved_view[(y) % 224][(x+ 2) % 240] = [0, 0, 255]
			percieved_view[(y) % 224][(x+ 3) % 240] = [0, 0, 255]
			percieved_view[(y) % 224][(x- 1) % 240] = [0, 0, 255]
			percieved_view[(y) % 224][(x- 2) % 240] = [0, 0, 255]
			percieved_view[(y) % 224][(x- 3) % 240] = [0, 0, 255]

			percieved_view[(y+ 1) % 224][(x) % 240] = [0, 0, 255]
			percieved_view[(y+ 2) % 224][(x) % 240] = [0, 0, 255]
			percieved_view[(y+ 3) % 224][(x) % 240] = [0, 0, 255]
			percieved_view[(y- 1) % 224][(x) % 240] = [0, 0, 255]
			percieved_view[(y- 2) % 224][(x) % 240] = [0, 0, 255]
			percieved_view[(y- 3) % 224][(x) % 240] = [0, 0, 255]

		[x, y] = position[0].type(torch.IntTensor).detach()

		percieved_view[(y) % 224][(x+ 1) % 240] = [0, 255, 255]
		percieved_view[(y) % 224][(x+ 2) % 240] = [0, 255, 255]
		percieved_view[(y) % 224][(x+ 3) % 240] = [0, 255, 255]
		percieved_view[(y) % 224][(x- 1) % 240] = [0, 255, 255]
		percieved_view[(y) % 224][(x- 2) % 240] = [0, 255, 255]
		percieved_view[(y) % 224][(x- 3) % 240] = [0, 255, 255]

		percieved_view[(y+ 1) % 224][(x) % 240] = [0, 255, 255]
		percieved_view[(y+ 2) % 224][(x) % 240] = [0, 255, 255]
		percieved_view[(y+ 3) % 224][(x) % 240] = [0, 255, 255]
		percieved_view[(y- 1) % 224][(x) % 240] = [0, 255, 255]
		percieved_view[(y- 2) % 224][(x) % 240] = [0, 255, 255]
		percieved_view[(y- 3) % 224][(x) % 240] = [0, 255, 255]

		game.imshow(percieved_view)

		self.enemies[self.batch] = enemies[0]
		self.user_actions = self.env.action_space.sample()
		# self.user_actions = prob[0].multinomial(1).view(9, 1)
		# self.actions.append(self.user_actions)
		# log_prob = prob.log()
		# entropy = -(log_prob * prob).sum(2).sum(1)
		# log_prob = torch.gather(log_prob[0], 1, self.user_actions)
		# self.log_probs.append(log_prob)
		# self.entropies.append(entropy)
		# self.values.append(value)
		

		if self.batch == batch_size:
			self.optimize()
		else:
			self.batch += 1




emulator = SAAEmulator()
emulator.run()
sys.exit(0)