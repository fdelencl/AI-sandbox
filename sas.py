from emulator import Emulator
from sequenced_analysis_attack.model import Model

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import numpy as np


act = torch.tensor([[1, 2]], dtype=torch.float)
act = F.softmax(act, dim=1)
print(act)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)
print(act.multinomial(1).data)



# class SASEmulator(Emulator):
# 	def __init__(self, fps=0):
# 		super(SASEmulator, self).__init__(fps=fps)




# emulator = SASEmulator()

# emulator.run()