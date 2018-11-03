#!/usr/bin/env python3
import retro
import signal
import time
import sys

class Emulator:

	def __init__(
		self,
		game='Gradius-Nes',
		state='Level1',
		scenario=None,
		info=None,
		render=True,
		fps=50):

		self.render = render
		self.fps = fps
		self.env = retro.make(game=game, state=state)
		self.env.reset()
		self.user_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.end = False
		if (render):
			self.env.render()
			self.env.unwrapped.viewer.window.on_key_press = self.on_key_press
			self.env.unwrapped.viewer.window.on_key_release = self.on_key_release

	def on_key_press(self, symbol, modifiers):
		# print(symbol)
		if symbol == 65505: #shift -> pick option
			self.user_actions[0] = 1
		if symbol == 119:   #w -> up
			self.user_actions[4] = 1
		if symbol == 115:   #s -> down
			self.user_actions[5] = 1
		if symbol == 97:    #a -> left
			self.user_actions[6] = 1
		if symbol == 100:   #d -> right
			self.user_actions[7] = 1
		if symbol == 32:    #spacebar -> shoot
			self.user_actions[8] = 1
		pass

	def on_key_release(self, symbol, modifiers):
		global  user_actions
		if symbol == 65288: #backspace
			self.env.reset()
		if symbol == 65505: #shift -> pick option
			self.user_actions[1] = 0
		if symbol == 119:   #w -> up
			self.user_actions[4] = 0
		if symbol == 115:   #s -> down
			self.user_actions[5] = 0
		if symbol == 97:    #a -> left
			self.user_actions[6] = 0
		if symbol == 100:   #d -> right
			self.user_actions[7] = 0
		if symbol == 32:    #spacebar -> shoot
			self.user_actions[8] = 0
		if symbol == 65307: #escape -> end_game
			self.end = True
		pass

	def before_run(self):
		return

	def run(self):
		self.before_run()
		skipticks = 1 / self.fps
		while not self.end:
			frame_start = time.perf_counter()
			self.step()
			frame_end = time.perf_counter()
			if frame_end > frame_start + skipticks:
				pass
			else:
				time.sleep(frame_start + skipticks - frame_end)

	def before_step(self):
		return

	def step(self):
		self.before_step()
		screen, _, done, _ = self.env.step(self.user_actions);
		if self.render:
			self.env.render()
		self.after_step()
		if done:
			self.done()


	def after_step(self):
		return

	def done(self):
		self.env.reset()

	def end(self):
		self.env.close()

if __name__ == "__main__":
	emul = Emulator(game='Gradius-Nes')
	emul.run()
