# File: run_agent.py
# Enhanced version with error handling for Q-table and manual time addition

# Importing necessary modules
from env import Environment
from agent_brain import QLearningTable
import matplotlib.pyplot as plt
import pygame          
import pandas as pd
import time  # Importing time module for manual time tracking

def plot_results(steps, all_costs):
    """Plotting the results - Steps per episode and Costs."""
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps', color=color)
    ax1.plot(steps, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Instantiate a second y-axis
    color = 'tab:red'
    ax2.set_ylabel('Costs', color=color)
    ax2.plot(all_costs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Avoid overlap
    plt.title('Steps and Costs over Episodes')
    plt.show()


def display_summary_table(data):
    """Generate and display a summary table of the agent's performance."""
    df = pd.DataFrame(data, columns=['Path', 'Steps', 'Cost', 'Time (seconds)'])
    print("\nPerformance Summary:\n")
    print(df)
    print(f"\nTotal Paths: {len(data)}")
    print(f"Average Steps: {df['Steps'].mean():.2f}")
    print(f"Average Cost: {df['Cost'].mean():.2f}")
    print(f"Total Time: {df['Time (seconds)'].sum():.2f} seconds")


def update():
    """Main loop to run episodes and update environment."""
    steps = []
    all_costs = []
    summary_data = []

    try:
        for episode in range(1000):
            observation = env.reset()
            i = 0
            cost = 0
            start_time = time.time()  # Start time using time module

            while True:
                env.render()
                action = RL.choose_action(str(observation))
                observation_, reward, done = env.step(action)
                cost += RL.learn(str(observation), action, reward, str(observation_))
                observation = observation_
                i += 1
                if done:
                    steps.append(i)
                    all_costs.append(cost)
                    end_time = time.time()  # End time using time module
                    summary_data.append([episode + 1, i, cost, end_time - start_time])  # Calculate elapsed time
                    break

        env.final()

        # Safely print the Q-table
        try:
            RL.print_q_table()
        except Exception as e:
            print(f"Error displaying Q-table: {e}")

        plot_results(steps, all_costs)
        print("\nTraining Completed Successfully!")
        display_summary_table(summary_data)

    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    env = Environment()  # Using default initialization of Environment class
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)  # Or just call update() for non-interactive environments
    env.mainloop()
    pygame.quit()

