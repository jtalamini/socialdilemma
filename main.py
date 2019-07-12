from agent import Agent
from prisoner import Prisoner

num_agents = 2
num_episodes = 1000000

game = Prisoner()
agents = [Agent(state_space=(1,), act_space=2, epsilon=1.0, gamma=0) for _ in range(num_agents)]

for episode in range(num_episodes):
    actions = []
    for agent in agents:
        actions.append(agent.choose_action(0))
    payoffs = game.play(actions)

    for i in range(len(agents)):
        agents[i].train((0,), actions[i], -payoffs[i], 0)
        agents[i].schedule_exploration()
    print("episode: ", episode, "actions: ",actions, "payoff:", payoffs, "epsilon: ", agents[0].epsilon)


