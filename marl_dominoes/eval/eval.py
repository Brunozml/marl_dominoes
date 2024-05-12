import numpy as np
from open_spiel.python.algorithms import random_agent

def eval_against_random_bots(env, trained_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    # random agents for evaluation
  num_actions = env.action_spec()["num_actions"]
  num_players = env.num_players
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
    
    # # Episode is over, step all agents with final info state.
    # for agent in random_agents:
    #     agent.step(time_step)
  
  return sum_episode_rewards / num_episodes
