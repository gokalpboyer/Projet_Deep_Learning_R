from monty_hall import MontyHallEnv
from value_iteration import ValueIterationAgent
from policy_iteration import PolicyIterationAgent
from monte_carlo_es import MonteCarloESAgent
from on_policy_first_visit_mc import OnPolicyFirstVisitMCAgent
from off_policy_mc import OffPolicyMCAgent
from dyna_q import DynaQAgent

env = MontyHallEnv()

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

# Monte Carlo ES
mc_es_agent = MonteCarloESAgent(env)
mc_es_agent.learn()
mc_es_policy = mc_es_agent.extract_policy()
print("Monte Carlo ES Policy:", mc_es_policy)

# On-policy First-visit MC
on_policy_mc_agent = OnPolicyFirstVisitMCAgent(env)
on_policy_mc_agent.learn()
on_policy_mc_policy = on_policy_mc_agent.extract_policy()
print("On-policy First Visit MC Policy:", on_policy_mc_policy)

# Off-policy MC
off_policy_mc_agent = OffPolicyMCAgent(env)
off_policy_mc_agent.learn()
off_policy_mc_policy = off_policy_mc_agent.extract_policy()
print("Off-policy MC Policy:", off_policy_mc_policy)

# Dyna-Q

dyna_q_agent = DynaQAgent(env)
dyna_q_agent.learn()
dyna_q_policy = dyna_q_agent.extract_policy()
print("Dyna-Q Policy:", dyna_q_policy)
