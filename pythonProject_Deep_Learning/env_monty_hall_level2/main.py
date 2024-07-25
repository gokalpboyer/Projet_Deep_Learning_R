from monty_hall_level2 import MontyHallEnvLevel2
from policy_iteration import PolicyIterationAgent
from value_iteration import ValueIterationAgent

env = MontyHallEnvLevel2()

# Value Iteration
vi_agent = ValueIterationAgent(env)
vi_agent.value_iteration()
vi_policy = vi_agent.extract_policy()
print("Value Iteration Policy:", vi_policy)

# Policy Iteration
pi_agent = PolicyIterationAgent(env)
pi_agent.policy_iteration()
pi_policy = pi_agent.extract_policy()
print("Policy Iteration Policy:", pi_policy)
