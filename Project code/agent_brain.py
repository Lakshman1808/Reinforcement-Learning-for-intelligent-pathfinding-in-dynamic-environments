# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import final_states  # Assuming final_states is defined in env.py

# Creating class for the Q-learning table
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        """Choose an action based on the current observation using epsilon-greedy strategy."""
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # Shuffle to break ties
            action = state_action.idxmax()  # Choose action with highest Q-value
        else:
            action = np.random.choice(self.actions)  # Choose random action
        return action

    def learn(self, state, action, reward, next_state):
        """Update the Q-value based on the action taken and the received reward."""
        self.check_state_exist(next_state)

        q_predict = self.q_table.loc[state, action]
        if next_state != 'goal' and next_state != 'obstacle':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()  # Bellman equation
        else:
            q_target = reward

        # Update the Q-value using the Q-learning update rule
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)
        return self.q_table.loc[state, action]

    def check_state_exist(self, state):
        """Check if the state exists in the Q-table, and append if it does not."""
        if state not in self.q_table.index:
            # Add a new row for the unseen state with all Q-values initialized to 0
            new_row = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])  # Append the new row to the Q-table

    def print_q_table(self):
        """Print the final Q-table."""
        try:
            e = final_states()  # Get the list of final states from the environment
            for state in e:
                if state in self.q_table.index:
                    self.q_table_final.loc[str(state), :] = self.q_table.loc[str(state), :]
                else:
                    print(f"State {state} is not in the Q-table. Skipping.")

            print(f'\nLength of final Q-table = {len(self.q_table_final.index)}')
            print('Final Q-table with values from the final route:')
            print(self.q_table_final)

            print(f'\nLength of full Q-table = {len(self.q_table.index)}')
            print('Full Q-table:')
            print(self.q_table)
        except Exception as err:
            print(f"Error while printing the Q-table: {err}")

    def plot_results(self, steps, cost):
        """Plot the results of the learning process."""
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        # Plot steps per episode
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via Steps')

        # Plot cost per episode
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via Cost')

        plt.tight_layout()
        plt.show()

