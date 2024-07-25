import os
import pickle
from environments.line_world import LineWorld
from algorithms.value_iteration import value_iteration
from algorithms.policy_iteration import policy_iteration
from algorithms.monte_carlo_es import monte_carlo_es
from algorithms.sarsa import sarsa


def main():
    env = LineWorld(length=5)

    print("Choose a mode:")
    print("1. Automated (Value Iteration)")
    print("2. Automated (Policy Iteration)")
    print("3. Automated (Monte Carlo ES)")
    print("4. Automated (Sarsa)")
    print("5. Manual")
    mode = input("Enter the number of the mode: ")

    if mode == "1":
        print("Value Iteration:")
        policy, value_table = value_iteration(env)
        print("Optimal Policy:", policy)
        print("Value Table:", value_table)
        save_policy(policy, 'saved_policies/value_iteration_policy.pkl')
        run_policy(env, policy)

    elif mode == "2":
        print("Policy Iteration:")
        policy, value_table = policy_iteration(env)
        print("Optimal Policy:", policy)
        print("Value Table:", value_table)
        save_policy(policy, 'saved_policies/policy_iteration_policy.pkl')
        run_policy(env, policy)

    elif mode == "3":
        print("Monte Carlo ES:")
        policy, Q = monte_carlo_es(env)
        print("Optimal Policy:", policy)
        save_policy(policy, 'saved_policies/monte_carlo_es_policy.pkl')
        run_policy(env, policy)

    elif mode == "4":
        print("Sarsa:")
        policy, Q = sarsa(env)
        print("Optimal Policy:", policy)
        save_policy(policy, 'saved_policies/sarsa_policy.pkl')
        run_policy(env, policy)

    elif mode == "5":
        run_policy_manual(env)


def run_policy(env, policy):
    state = env.reset()
    env.render()
    while True:
        action = policy[state]
        state, reward, done = env.step(action)
        env.render()
        if done:
            break


def run_policy_manual(env):
    state = env.reset()
    env.render()
    print("Enter actions (0 for left, 1 for right). Type 'exit' to stop.")
    while True:
        action = input("Action: ")
        if action == "exit":
            break
        action = int(action)
        state, reward, done = env.step(action)
        env.render()
        if done:
            print("Reached the goal!")
            break


def save_policy(policy, filename):
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(policy, f)


def load_policy(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
