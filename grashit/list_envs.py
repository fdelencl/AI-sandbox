import retro

for game in retro.list_games():
    print(game, retro.list_states(game))