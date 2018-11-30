import pickle
import json
import time
import os

class Reader:

	def __init__(self, index_path='./memory_pipe/memory/db_index.json', db_path='./memory_pipe/memory'):
		self.db_path = db_path
		self.index_path = index_path
		if os.path.isfile(index_path):
			with open(index_path, 'r') as index_fd:
				self.index = json.load(index_fd)
				self.session_index = 0
				self.current_session = self.index[self.session_index]
				self.current_frame = 0
				self.open_session()
		else:
			print('could not load index at', index_path)

	def open_session(self, session=None):
		if session:
			self.session_index = session
			self.current_session = self.index[self.session_index]
		if os.path.isfile(self.current_session['filepath']):
			with open(self.current_session['filepath'], 'rb') as session_fd:
				self.data = pickle.load(session_fd)
				self.current_frame = 0

	def next_frame(self):
		if not self.data or self.current_frame >= self.data['length']:
			self.open_session((self.session_index + 1) % len(self.index))
			self.current_frame = 0
		frame = {
			'RAM': self.data['RAMs'][self.current_frame],
			'input': self.data['inputs'][self.current_frame]
		}
		self.current_frame += 1
		return frame
