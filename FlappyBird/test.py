import flappy
import random


obs = flappy.main()

while True:
    action = random.randint(0, 1)
    obs, reward = flappy.game_step(action, obs)
    #print(obs['dead'])
    if obs['dead'] == True:
        movementInfo = flappy.showWelcomeAnimation()
        obs = flappy.mainGame(movementInfo)
