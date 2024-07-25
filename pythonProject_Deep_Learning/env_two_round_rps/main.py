from two_round_rps import TwoRoundRPS
from value_iteration import ValueIterationAgent
from policy_iteration import PolicyIterationAgent
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from monte_carlo_es import MonteCarloES
from on_policy_first_visit_mc import OnPolicyFirstVisitMC
from off_policy_mc import OffPolicyMC
from dyna_q import DynaQAgent

if __name__ == "__main__":
    env = TwoRoundRPS()

    # Value Iteration
    vi_agent = ValueIterationAgent(env)
    vi_agent.value_iteration()
    print("Value Iteration Policy:", vi_agent.extract_policy())

    # Policy Iteration
    pi_agent = PolicyIterationAgent(env)
    pi_agent.train()
    print("Policy Iteration Policy:", pi_agent.extract_policy())

    # Q-Learning
    ql_agent = QLearningAgent(env)
    ql_agent.learn()
    print("Q-Learning Policy:", ql_agent.extract_policy())

    # SARSA
    sarsa_agent = SARSAAgent(env)
    sarsa_agent.learn()
    print("SARSA Policy:", sarsa_agent.extract_policy())

    # Monte Carlo ES
    mc_es_agent = MonteCarloES(env, episodes=1000)
    mc_es_agent.learn()
    print("Monte Carlo ES Policy:", mc_es_agent.extract_policy())

    # On-policy First Visit MC
    on_policy_mc_agent = OnPolicyFirstVisitMC(env, episodes=1000)
    on_policy_mc_agent.learn()
    print("On-policy First Visit MC Policy:", on_policy_mc_agent.extract_policy())

    # Off-policy MC
    off_policy_mc_agent = OffPolicyMC(env, episodes=1000)
    off_policy_mc_agent.learn()
    print("Off-policy MC Policy:", off_policy_mc_agent.extract_policy())

    # Dyna-Q
    dyna_q_agent = DynaQAgent(env, episodes=1000)
    dyna_q_agent.learn()
    print("Dyna-Q Policy:", dyna_q_agent.extract_policy())
