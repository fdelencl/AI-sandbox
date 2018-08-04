import retro


for game in retro.data.list_games():
	print(game, retro.data.list_states(game))

env = retro.make(game='Gradius-Nes', state='Level1', record='.')
env.reset()
done = False

# [??, B, Select, Start, up, down, left, right, A]

while done == False:
	actions = [0, 0, 0, 0, 0, 0, 0, 0, 1]
	print('actions', actions)
	_obs, _rew, done, _info = env.step(actions)
	print('_obs', _obs.shape)
	print('_rew', _rew)
	print('done', done)
	print('_info', _info)
	env.render()

print('what the fuck')
