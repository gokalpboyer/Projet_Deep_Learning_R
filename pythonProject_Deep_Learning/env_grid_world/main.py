import numpy as np
from grid_world import GridWorld
from value_iteration import ValueIterationAgent
from policy_iteration import PolicyIterationAgent
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from monte_carlo_es import MonteCarloES
from on_policy_first_visit_mc import OnPolicyFirstVisitMC
from off_policy_mc import OffPolicyMC
from dyna_q import DynaQAgent

def display_policy(policy, env):
    directions = ['↑', '↓', '←', '→']
    action_to_index = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}

    width, height = env.width, env.height
    start = env.start
    terminal_states = env.terminal_states
    obstacles = env.obstacles

    for i in range(height):
        for j in range(width):
            if (i, j) in terminal_states:
                print('T', end=' ')  # Représentation des états terminaux
            elif (i, j) in obstacles:
                print('X', end=' ')  # Représentation des obstacles
            elif (i, j) == start:
                print('S', end=' ')  # Représentation du point de départ
            else:
                action_index = None
                if isinstance(policy, dict):
                    action = policy.get((i, j))
                    if isinstance(action, str):
                        action_index = action_to_index.get(action)
                    elif isinstance(action, int):
                        action_index = action
                elif isinstance(policy, np.ndarray):
                    action_index = policy[i, j] if policy[i, j] != -1 else None

                if action_index is not None and 0 <= action_index < len(directions):
                    print(directions[action_index], end=' ')
                else:
                    print(' ', end=' ')  # Affichage d'un espace vide pour les états sans action
        print()
    print()

def display_values(values, env):
    width, height = env.width, env.height
    for i in range(height):
        for j in range(width):
            value = values.get((i, j), None)
            if value is not None:
                print(f"{value:.2f}", end=' ')
            else:
                if (i, j) in env.terminal_states:
                    print('T', end=' ')
                elif (i, j) in env.obstacles:
                    print('X', end=' ')
                elif (i, j) == env.start:
                    print('S', end=' ')
                else:
                    print('0.00', end=' ')  # Affichage de 0.00 pour les valeurs non définies
        print()
    print()

if __name__ == "__main__":
    width, height = 4, 4
    start = (0, 0)
    terminal_states = [(3, 3)]
    obstacles = [(1, 1)]

    env = GridWorld(width=width, height=height, start=start, terminal_states=terminal_states, obstacles=obstacles)

    # Value Iteration
    vi_agent = ValueIterationAgent(env)
    vi_agent.value_iteration()
    vi_policy = vi_agent.extract_policy()
    vi_values = {state: value for state, value in np.ndenumerate(vi_agent.value_function)}
    print("Value Iteration Policy:")
    display_policy(vi_policy, env)
    print("Value Iteration Values:")
    display_values(vi_values, env)

    # Policy Iteration
    pi_agent = PolicyIterationAgent(env)
    pi_agent.train()
    pi_policy = pi_agent.extract_policy()
    pi_values = {state: value for state, value in np.ndenumerate(pi_agent.value_function)}
    print("Policy Iteration Policy:")
    display_policy(pi_policy, env)
    print("Policy Iteration Values:")
    display_values(pi_values, env)

    # Q-Learning
    ql_agent = QLearningAgent(env)
    ql_agent.learn()
    ql_policy = ql_agent.extract_policy()
    ql_values = {state: max(actions.values()) for state, actions in ql_agent.q_table.items()}
    print("Q-Learning Policy:")
    display_policy(ql_policy, env)
    print("Q-Learning Values:")
    display_values(ql_values, env)

    # SARSA
    sarsa_agent = SARSAAgent(env)
    sarsa_agent.learn()
    sarsa_policy = sarsa_agent.extract_policy()
    sarsa_values = {(x, y): np.max(sarsa_agent.q_table[x, y]) for x in range(env.height) for y in range(env.width) if (x, y) not in env.terminal_states and (x, y) not in env.obstacles}
    print("SARSA Policy:")
    display_policy(sarsa_policy, env)
    print("SARSA Values:")
    display_values(sarsa_values, env)

    # Monte Carlo ES
    mc_es_agent = MonteCarloES(env)
    mc_es_agent.learn()
    mc_es_policy = mc_es_agent.extract_policy()
    mc_es_values = {(x, y): max(mc_es_agent.Q[(x, y)].values()) if (x, y) in mc_es_agent.Q else None for x in range(env.height) for y in range(env.width)}
    print("Monte Carlo ES Policy:")
    display_policy(mc_es_policy, env)
    print("Monte Carlo ES Values:")
    display_values(mc_es_values, env)

    # On-Policy First-Visit MC
    on_policy_mc_agent = OnPolicyFirstVisitMC(env)
    on_policy_mc_agent.learn()
    on_policy_mc_policy = on_policy_mc_agent.extract_policy()
    on_policy_mc_values = {(x, y): max(on_policy_mc_agent.Q[(x, y)].values()) if (x, y) in on_policy_mc_agent.Q else None for x in range(env.height) for y in range(env.width)}
    print("On-Policy First-Visit MC Policy:")
    display_policy(on_policy_mc_policy, env)
    print("On-Policy First-Visit MC Values:")
    display_values(on_policy_mc_values, env)

    # Off-Policy MC
    off_policy_mc_agent = OffPolicyMC(env)
    off_policy_mc_agent.learn()
    off_policy_mc_policy = off_policy_mc_agent.extract_policy()
    off_policy_mc_values = {(x, y): max(off_policy_mc_agent.Q[(x, y)].values()) if (x, y) in off_policy_mc_agent.Q else None for x in range(env.height) for y in range(env.width)}
    print("Off-Policy MC Policy:")
    display_policy(off_policy_mc_policy, env)
    print("Off-Policy MC Values:")
    display_values(off_policy_mc_values, env)

    # Dyna-Q
    dyna_q_agent = DynaQAgent(env)
    dyna_q_agent.learn()
    dyna_q_policy = dyna_q_agent.extract_policy()
    dyna_q_values = {state: max(actions.values()) for state, actions in dyna_q_agent.Q.items()}
    print("Dyna-Q Policy:")
    display_policy(dyna_q_policy, env)
    print("Dyna-Q Values:")
    display_values(dyna_q_values, env)
