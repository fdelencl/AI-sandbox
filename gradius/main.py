import argparse
import retro
import time
import torch.optim as optim
import signal
import sys

import numpy as np

from model import Net
from optimizer import SharedRMSprop

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--human',
    type=bool,
    default=True,
    help='to play with keyboard control')

env = retro.make(game='Gradius-Nes', state='Level1')
# env = retro.make(game='GradiusIII-Snes', state='Level1.Mode1.Shield')

stop = False
def signal_handler(sig = None, frame = None):
	global stop
	stop = True
signal.signal(signal.SIGINT, signal_handler)

done = False

user_actions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

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

screen = env.reset()
env.render()

env.unwrapped.viewer.window.on_key_press = on_key_press
env.unwrapped.viewer.window.on_key_release = on_key_release

fps = 50
skipticks = 1/(fps*1.0)

shared_model = Net(env.observation_space.shape[0], user_actions)
optimizer = SharedRMSprop(shared_model.parameters(), lr=0.001)
hx = None
cx = None
def loop():
	global screen, hx, cx
	cr_lin, action, stuff, (hx, cx) = shared_model.select_action(screen, (hx, cx))
	print(action)
	screen, _rew, done, _info = env.step(action)
	env.render()

if __name__ == '__main__':
	args = parser.parse_args()

	# shared_model = Net(env.observation_space.shape[0], user_actions)
	# optimizer = SharedRMSprop(shared_model.parameters(), lr=0.001)

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