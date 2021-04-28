from gym.envs.registration import register

register(
    id='flap-v0',
    entry_point='IntelliFlap.envs:FlapEnv',
)

register(
    id='cnnflap-v0',
    entry_point='IntelliFlap.envs:CnnFlapEnv',
)
