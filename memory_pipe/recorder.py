import pickle
import json
import time
import os

class Recorder:

	def __init__(self, index_path='./memory_pipe/memory/db_index', db_path='./memory_pipe/memory'):
		self.db_path = db_path
		self.index_path = index_path
		if os.path.isfile(index_path):
			with open(index_path, 'r') as index_fd:
				self.index = json.load(index_fd)
		else:
			self.index = []
		self.new_session()

	def new_session(self, player=None):
		self.session = { 'RAMs': [], 'inputs': [], 'screens': [], 'recorded_at': time.time(), 'player': player}

	def record(self, RAM, input, screen):
		self.session['RAMs'].append(RAM)
		self.session['inputs'].append(input)
		self.session['screens'].append(screen)

	def save_session(self, final_score=0):
		self.session['length'] = len(self.session['RAMs'])
		self.session['duration'] = time.time() - self.session['recorded_at']
		session_index = len(self.index)
		session_path = "%s/session%d.dump" % (self.db_path, session_index)
		i = {
			'recorded_at': self.session['recorded_at'],
			'duration': self.session['duration'],
			'length': self.session['length'],
			'filepath': session_path,
			'final_score': final_score,
			'player': self.session['player']
		}
		self.index.append(i)
		with open(session_path, 'wb') as session_fd:
			pickle.dump(self.session, session_fd)
		with open(self.index_path, 'w') as index_fd:
			self.index = json.dump(self.index, index_fd)
		self.new_session()
