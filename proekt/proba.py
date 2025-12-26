import pettingzoo.classic.tictactoe_v3 as ttt
import time
import numpy as np
env = ttt.env(render_mode="human")
env.reset()

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = obs["action_mask"]
        legal_actions = np.flatnonzero(mask)
        action = np.random.choice(legal_actions)

    env.step(action)
    env.render()
    time.sleep(0.6)

print("Render finished")
