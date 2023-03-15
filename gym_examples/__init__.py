from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id="gym_examples/StarWars-v0",
    entry_point="gym_examples.envs:StarWarsEnv",
    max_episode_steps=300,
)

register(
    id="gym_examples/CNN-v0",
    entry_point="gym_examples.envs:CNNEnv",
    max_episode_steps=300,
)