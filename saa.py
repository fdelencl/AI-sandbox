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

SAA = Model().to(device)
model_path = './rom/saa.model'
SAA.load(model_path)

current_score = 0

class SAAEmulator(Emulator):
	def __init__(self, fps=0):
		super(SAAEmulator, self).__init__(fps=fps)

	def done(self)
		global model_path
		current_score = 0
		SAA.save(model_path)
		super(MyEmulator, self).done()

	def before_step(self):
		self.user_actions = self.action_spectrum[index]

	def after_step(self, RAM, input, screen, info):
		global current_score
		delta_score = info['score'] - current_score
		reward = 1 + delta_score


emulator = SAAEmulator()
emulator.run()
sys.exit(0)