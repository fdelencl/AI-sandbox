import retro

env = retro.make(game='Gradius-Nes', state='Level1', record='.')
env.reset()
done = False

# [??, B, Select, Start, up, down, left, right, A]

while done == False:
        actions = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        _obs, _rew, done, _info = env.step(actions)
        env.render()
        
print('what the fuck')
