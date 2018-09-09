#!/usr/local/bin/python3

import retro

from net import Net, device
from memory import Memory
import time
import torch.optim as optim

import signal
import sys

env = retro.make(game='Gradius-Nes', state='Level1')
# env = retro.make(game='GradiusIII-Snes', state='Level1.Mode1.Shield')

# [??, B, Select, Start, up, down, left, right, A]

n = Net().to(device)
m = Memory(15000)
o = optim.RMSprop(policy_net.parameters())


# m.load('loli.kitty')

stop = False
def signal_handler(sig, frame):
	global stop
	stop = True
signal.signal(signal.SIGINT, signal_handler)

done = False

user_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0]

def on_key_press(symbol, modifiers):
	print(symbol, 'pressed')
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
	print(symbol, 'released')
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
	pass

current_screen = env.reset()
last_screen = current_screen
state = current_screen - last_screen
env.render()

env.unwrapped.viewer.window.on_key_press = on_key_press
env.unwrapped.viewer.window.on_key_release = on_key_release

def optimize():


def loop():
	global current_screen, last_screen, state
	last_screen = current_screen
	action = user_actions
	current_screen, _rew, done, _info = env.step(action)
	if not done:
		next_state = current_screen - last_screen
	else:
		next_state = None
		env.reset()
	m.push(state, action, next_state, _rew)
	state = next_state
	env.render()

fps = 50
skipticks = 1/(fps*1.0)
while not stop:
	tim1 = time.perf_counter()
	loop()
	tim2 = time.perf_counter()
	if tim2 > tim1 + skipticks:
		print('took already too long')
	else:
		time.sleep(tim1 + skipticks - tim2)




m.save('loli')
env.close()
sys.exit(0)