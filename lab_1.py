"""Reinformcent Learning - Lab 1

For being able to watch the videos, I've installed the ffmpeg library via `sudo apt install ffmpeg`.

We will work with “The Cliff”. Some definitions (see [Cliff Walking - Gymnasium Documentation](https://gymnasium.farama.org/environments/toy_text/cliff_walking/))
- state: The world is a $4\times12$ grid (you should rotate your head when you see the map, so as to see it extending vertically instead of horizontally), the player starts at $[3,0]$ and the goal is at $[3,11]$, with a cliff that runs along $[3, 1 \text{ to } 10]$. There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at the goal as the latter results in the end of the episode. The posible states are maped to a single integer (instead of a tuple) by doing `current_row * nrows + current_col` (where both the row and col start at 0). E.g. the stating position can be calculated as follows: 3 * 12 + 0 = 36.
- actions
    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left
- q: `dict` with `(state, a)` keys and `float` values (that stand for the value of taking that action in that state).
- hyperparameters: `dict` defined below, with the necessary values for SARSA and Q-learning.

"""


import itertools
from typing import Any, Callable, Tuple, List
from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
import imageio


# -------------------- Some plotting functions -------------------------------


def plot_reward_per_episode(reward_ep) -> None:
    episode_rewards = np.array(reward_ep)

    # se suaviza la curva de convergencia
    episode_number = np.linspace(1, len(episode_rewards) + 1, len(episode_rewards) + 1)
    acumulated_rewards = np.cumsum(episode_rewards)

    reward_per_episode = [
        acumulated_rewards[i] / episode_number[i]
        for i in range(len(acumulated_rewards))
    ]

    plt.plot(reward_per_episode)
    plt.title("Recompensa acumulada por episodio")
    # plt.show()
    # we save the plot
    plt.savefig("reward_per_episode.png", dpi=1000)


def plot_steps_per_episode(timesteps_ep) -> None:
    # se muestra la curva de aprendizaje de los pasos por episodio
    episode_steps = np.array(timesteps_ep)
    plt.plot(np.array(range(0, len(episode_steps))), episode_steps)
    plt.title("Pasos (timesteps) por episodio")
    # plt.show()
    # we save the plot
    plt.savefig("steps_per_episode.png", dpi=1000)


def plot_steps_per_episode_smooth(timesteps_ep) -> None:
    episode_steps = np.array(timesteps_ep)

    # se suaviza la curva de aprendizaje
    episode_number = np.linspace(1, len(episode_steps) + 1, len(episode_steps) + 1)
    acumulated_steps = np.cumsum(episode_steps)

    steps_per_episode = [
        acumulated_steps[i] / episode_number[i] for i in range(len(acumulated_steps))
    ]

    plt.plot(steps_per_episode)
    plt.title("Pasos (timesteps) acumulados por episodio")
    # plt.show()
    # we save the plot
    plt.savefig("steps_per_episode_smooth.png", dpi=1000)
    plt.close()


# This function combines the three previous ones and has a better output.
def plot_combined_metrics(
    reward_ep: list,
    timesteps_ep: list,
    hyperparameters: dict = None,
    include_rewards: bool = True,
    include_timesteps: bool = True,
) -> None:
    """
    Plot the reward per episode and/or the number of timesteps per episode.

    Parameters
    ----------
    reward_ep : list
        List containing the reward obtained per episode.
    timesteps_ep : list
        List containing the number of timesteps taken per episode.
    hyperparameters : dict, optional
        Dictionary containing hyperparameters like alpha, gamma, epsilon, and tau.
    include_rewards : bool, optional
        Flag to include reward metrics in the plot.
    include_timesteps : bool, optional
        Flag to include timestep metrics in the plot.

    Returns
    -------
    None
    """

    episode_number = np.linspace(1, len(timesteps_ep) + 1, len(timesteps_ep) + 1)

    # If include_rewards is True, plot reward per episode
    if include_rewards:
        episode_rewards = np.array(reward_ep)
        cum_rewards = np.cumsum(episode_rewards)
        reward_per_episode_smooth = [
            cum_rewards[i] / episode_number[i] for i in range(len(cum_rewards))
        ]
        plt.plot(reward_per_episode_smooth, label="Reward Smooth")
        plt.plot(reward_ep, label="Reward Original", alpha=0.5)

    # If include_timesteps is True, plot timesteps per episode
    if include_timesteps:
        episode_steps = np.array(timesteps_ep)
        cum_steps = np.cumsum(episode_steps)
        steps_per_episode_smooth = [
            cum_steps[i] / episode_number[i] for i in range(len(cum_steps))
        ]
        plt.plot(steps_per_episode_smooth, label="Timesteps Smooth")
        plt.plot(timesteps_ep, label="Timesteps Original", alpha=0.5)

    # Labels and titles
    plt.title("Metrics per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Metric Value")
    plt.legend()

    # File saving
    filename = "metrics_per_episode"
    if hyperparameters:
        filename += f"_alpha_{hyperparameters.get('alpha', 'NA')}_gamma_{hyperparameters.get('gamma', 'NA')}_epsilon_{hyperparameters.get('epsilon', 'NA')}_tau_{hyperparameters.get('tau', 'NA')}"
    filename += ".png"

    plt.savefig(filename, dpi=1000)
    plt.close()


def draw_value_matrix(q: dict) -> None:
    """Plots the action-value matrix based on a given Q-table.

    This function generates a heatmap to represent the value of the best
    action for each state in the gridworld “The Cliff” of OpenAI's `gymnasium`.
    It also annotates the heatmap with letters indicating the direction of the
    best action for each state.

    The function saves the generated plot as 'value_matrix.png'.

    Parameters
    ----------
    q : dict
        A dictionary mapping state-action pairs to Q-values. The state is
        represented as an integer, while the action is also represented as an
        integer. For example, `q[(0, 1)]` gives the Q-value for being in state
        0 and taking action 1.

    Returns
    -------
    None

    Notes
    -----
    - The function expects a 4x12 gridworld represented in a Q-table,
    consistent with OpenAI Gymnasium's CliffWalking environment.
    - The function uses matplotlib to generate the plot and saves it as a PNG
    file.
    - The terminal state (goal state) is hardcoded as state 47, corresponding
    to the position (3, 11) in the grid.
    """

    n_rows, n_columns, n_actions = 4, 12, 4
    q_value_matrix = np.empty((n_rows, n_columns))

    for row in range(n_rows):
        for column in range(n_columns):
            state = row * n_columns + column

            # Handle the terminal state explicitly
            if state == 47:
                q_value_matrix[row, column] = -1
                continue

            state_values = [q.get((state, action), -100) for action in range(n_actions)]
            q_value_matrix[row, column] = max(state_values)

    plt.imshow(q_value_matrix, cmap=plt.cm.RdYlGn)
    plt.colorbar()
    plt.title("Valor de la mejor acción en cada estado")

    for row, column in itertools.product(range(n_rows), range(n_columns)):
        state = row * n_columns + column
        best_action_value = -float("inf")
        best_action = None
        arrow_direction = ""

        # we map action indices to arrow directions
        action_map = {0: "U", 1: "R", 2: "D", 3: "L"}

        for action in range(n_actions):
            action_value = q.get((state, action), -1000)
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action_map[action]

        # we handle the terminal state explicitly
        if state == 47:
            best_action = ""

        plt.text(column, row, best_action, horizontalalignment="center")

    plt.xticks([])
    plt.yticks([])
    plt.savefig("value_matrix.png")
    # plt.show()
    plt.close()


# ------------------------ Policies ------------------------------------------


def choose_action_e_greedy(
    state: int,
    actions: range,
    q: dict,
    hyperparameters: dict,
    random_state: np.random.RandomState,
) -> int:
    """Choose an action according to an epsilon-greedy policy based on the 
    Q-values and epsilon parameter.
    
    Parameters
    ----------
    state : int
        The current state of the environment.
    actions : range
        The set of possible actions.
    q : dict
        A dictionary containing the Q-values for state-action pairs.
        The keys are tuples (state, action), and the values are the Q-values.
    hyperparameters : dict
        A dictionary containing hyperparameters for the algorithm.
        Must contain the key 'epsilon' for the epsilon-greedy policy.
    random_state : np.random.RandomState
        A random state object for reproducible randomness.
    
    Returns
    -------
    int
        The action selected according to the epsilon-greedy policy.
    
    Notes
    -----
    The epsilon-greedy policy works as follows:
        - With probability epsilon, choose a random action.
        - Otherwise, choose the action with the highest Q-value.
        If there are multiple actions with the same highest Q-value,
        one of them is selected randomly.
    """
    # ej: para 4 acciones inicializa en [0,0,0,0]
    q_values = [q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    # sorteamos un número: es menor a épsilon?
    if random_state.uniform() < hyperparameters["epsilon"]:
        # sí: se selecciona una acción aleatoria
        return random_state.choice(actions)

    count = q_values.count(max_q)

    # hay más de un máximo valor de estado-acción?
    if count > 1:
        # sí: seleccionamos uno de ellos aleatoriamente
        best = [i for i in range(len(actions)) if q_values[i] == max_q]
        i = random_state.choice(best)
    else:
        # no: seleccionamos el máximo valor de estado-acción
        i = q_values.index(max_q)

    return actions[i]


def choose_action_softmax(
    state: int,
    actions: range,
    q: dict,
    hyperparameters: dict,
    random_state: np.random.RandomState,
) -> int:
    """Choose an action according to a softmax policy, based on the
    Q-values and temperature parameter.
    
    Parameters
    ----------
    state : int
        The current state of the environment.
    actions : range
        The set of possible actions.
    q : dict
        A dictionary containing the Q-values for state-action pairs.
        The keys are tuples (state, action), and the values are the Q-values.
    hyperparameters : dict
        A dictionary containing hyperparameters for the algorithm.
        Must contain the key 'tau' for temperature.
    random_state : np.random.RandomState
        A random state object for reproducible randomness.
    
    Returns
    -------
    int
        The action selected according to the softmax policy.
    
    Notes
    -----
    The softmax policy is defined as:
        policy(a) = exp(Q(s, a) / tau) / Z
    where:
        Q(s, a) is the Q-value for state `s` and action `a`.
        tau is the temperature parameter controlling exploration.
        Z is the normalization constant.
    """
    # we get the value of each action in this state
    q_values = [q.get((state, a), 0.0) for a in actions]

    # we get the temperature to be used
    tau = hyperparameters["tau"]

    # we calculate the probabilities of the policy
    policy = [np.exp(q_values[a] / tau) for a in actions]
    norm = sum(policy)
    policy = [p / norm for p in policy]

    # now, we select action i with probability policy[i]
    # (we get a random number and then compare it with
    # the cumulative sum of the probabilities)
    rand_value = random_state.uniform()
    i = 0
    while rand_value > 0:
        rand_value -= policy[i]
        i += 1
    # note that i is incremented one more time than needed

    return actions[i - 1]



# ------------------ Action-value Learning algorithms ------------------------


def learn_SARSA(
    state: int,
    action: int,
    reward: int,
    next_state: int,
    next_action: int,
    q: dict,
    hyperparameters: dict,
) -> None:
    """Update the Q-value for the given state and action pair using SARSA
    learning (on-policy TD control).

    Parameters
    ----------
    state : int
        The current state.
    action : int
        The current action.
    reward : int
        The reward obtained by taking action `action` in state `state`.
    next_state : int
        The next state.
    next_action : int
        The next action.
    q : dict
        The Q-value table, represented as a dictionary with state-action pairs
        as keys and Q-values as values.
    hyperparameters : dict
        A dictionary of hyperparameters needed for the Q-value update.
        Expected to contain 'alpha' and 'gamma'.

    Notes
    -----
    - The function initializes the Q-value to 0.0 for new state-action pairs
    if they are not already present in `q`.
    - The Q-value for the current state-action pair is updated using the
    formula:

    .. math::
        Q(s, a) \leftarrow Q(s, a) + \alpha \times (r + \gamma \times Q(s', a') - Q(s, a))

    Where:
    - \( \alpha \) is the learning rate
    - \( \gamma \) is the discount factor
    - \( r \) is the reward
    - \( Q(s, a) \) is the current Q-value
    - \( Q(s', a') \) is the Q-value for the next state-action pair

    """

    # initialize the Q-value for brand-new state-action pairs
    if (state, action) not in q:
        q[(state, action)] = 0.0  # or another initial value
    if (next_state, next_action) not in q:
        q[(next_state, next_action)] = 0.0  # or another initial value

    # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
    q[(state, action)] += hyperparameters["alpha"] * (
        reward
        + hyperparameters["gamma"] * q[(next_state, next_action)]
        - q[(state, action)]
    )


def learn_Q_learning(
    state: int,
    action: int,
    reward: int,
    next_state: int,
    next_action: int,
    q: dict,
    hyperparameters: dict,
) -> None:
    """Update the Q-value for a given state-action pair using the Q-Learning
    algorithm.

    The function performs an off-policy Temporal Difference (TD) control to
    update the Q-value for a specific state and action based on the provided
    reward and the estimated optimal future value.

    Parameters
    ----------
    state : int
        The current state in the environment, represented as an integer.
    action : int
        The action taken in the current state, represented as an integer.
    reward : int
        The immediate reward received after taking the specified action in the
        given state.
    next_state : int
        The state transitioned to after taking the specified action.
    next_action : int
        The action to be taken in the next state.
    q : dict
        A dictionary mapping from state-action tuples to Q-values.
    hyperparameters : dict
        A dictionary containing the learning rate (alpha) and discount factor
        (gamma).

    Returns
    -------
    None

    Notes
    -----
    - The `next_action` parameter is included for signature consistency with
    the SARSA function, although it is not used.
    - The function expects the action space to be of size 4, which needs to be
    considered if adapting the code for other problems.
    - The function initializes the Q-value to 0.0 for new state-action pairs.
    - The Q-value for the current state-action pair is updated using the
    formula:

    .. math::
        Q(s, a) \leftarrow Q(s, a) + \alpha \times (r + \gamma \times \max_{a'} Q(s', a') - Q(s, a))

    Where:
    - \( \alpha \) is the learning rate
    - \( \gamma \) is the discount factor
    - \( r \) is the reward
    - \( Q(s, a) \) is the current Q-value
    - \( Q(s', a') \) is the Q-value for the next state and the action that
    maximizes it

    """

    # initialize the Q-value for brand-new state-action pairs
    if (state, action) not in q:
        q[(state, action)] = 0.0  # or another initial value
    if (next_state, next_action) not in q:
        q[(next_state, next_action)] = 0.0  # or another initial value

    # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
    q_max = max([q.get((next_state, a), 0.0) for a in range(4)])
    q[(state, action)] += hyperparameters["alpha"] * (
        reward + hyperparameters["gamma"] * q_max - q[(state, action)]
    )


# ------------------ Training Loop ------------------------


def run(
    learning_function: Callable,
    hyperparameters: dict,
    episodes_to_run: int,
    env: gym.Env,
    actions: range,
    q: dict,
    random_state: np.random.RandomState,
    render: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Corre el algoritmo de RL para el ambiente env.
    Args:
        learning_function: función de actualización de algoritmo de aprendizaje
        hyperparameters: hiperparámetros del algoritmo de aprendizaje
        episodes_to_run: cantidad de episodios a ejecutar
        env: ambiente de Gymnasium
        actions: lista de acciones posibles
        q: diccionario de valores de estado-acción
        random_state: generador de números aleatorios
    """
    # registro de la cantidad de pasos que le llevó en cada episodio
    # llegar a la salida
    timesteps_of_episode = []
    # cantidad de recompensa que recibió el agente en cada episodio
    reward_of_episode = []

    all_frames = []  # For storing frames of all episodes
    frames = []  # For storing frames of the current episode

    for episode in range(episodes_to_run):
        # se ejecuta una instancia del agente hasta que el mismo
        # llega a la salida o tarda más de 2000 pasos

        # reinicia el ambiente, obteniendo el estado inicial del mismo
        state, _ = env.reset()

        episode_reward = 0
        done = False
        t = 0

        # elige una acción basado en el estado actual.
        # Filtra el primer elemento de state, que es el estado en sí mismo
        action = choose_action_with_policy(
            state, actions, q, hyperparameters, random_state
        )

        while not done:
            # el agente ejecuta la acción elegida y obtiene los resultados
            next_state, reward, terminated, truncated, _ = env.step(action)

            if render:
                frame = env.render()
                all_frames.append(frame)
                if episode % 100 == 0:
                    frames.append(frame)

            next_action = choose_action_with_policy(
                next_state, actions, q, hyperparameters, random_state
            )

            episode_reward += reward
            learning_function(
                state, action, reward, next_state, next_action, q, hyperparameters
            )

            done = terminated or truncated

            # if the algorithm does not converge, it stops after 2000 timesteps
            if not done and t < 2000:
                state = next_state
                action = next_action
            else:
                # el algoritmo no ha podido llegar a la meta antes de dar 2000 pasos
                done = True  # se establece manualmente la bandera done
                timesteps_of_episode = np.append(timesteps_of_episode, [int(t + 1)])
                reward_of_episode = np.append(
                    reward_of_episode, max(episode_reward, -100)
                )

            t += 1

        # Create and save GIF after each 100 episodes
        if render and episode % 100 == 0:
            imageio.mimsave(f"episode_{episode}.gif", frames, duration=12)
            frames = []

    # Create a GIF for all episodes to show the learning process
    if render:
        imageio.mimsave("all_episodes.gif", all_frames, duration=12)

    return reward_of_episode.mean(), timesteps_of_episode, reward_of_episode


# =============== CUSTOM CODE FOR ANALYZING CONVERGENCE ======================


def run_multiple_experiments(
    learning_function,
    hyperparameters_list: List[dict],
    episodes_to_run: int,
    env,
    actions: List[int],
    q: dict,
    random_state: int,
    # render: bool = False,
) -> List[Tuple[list, list]]:
    """Run experiments with different sets of hyperparameters.

    Parameters
    ----------
    (All the parameters of the `run` function)
    hyperparameters_list : List[dict]
        List of dictionaries, each containing a set of hyperparameters.

    Returns
    -------
    List[Tuple[list, list]]
        A list of tuples, each containing reward_ep and timesteps_ep for each hyperparameter set.
    """

    render=False
    # this is hardcoded cause render=True doesn't make sense in this context,
    # buy I want to reuse the code of the `run` function.

    results = []

    for hyperparameters in hyperparameters_list:
        avg_rew_per_episode, timesteps_ep, reward_ep = run(
            learning_function,
            hyperparameters,
            episodes_to_run,
            env,
            actions,
            q,
            random_state,
            render=render,
        )
        results.append((reward_ep, timesteps_ep))

    return results


def plot_multiple_metrics(
    results: List[Tuple[list, list]], 
    hyperparameters_list: List[dict] = None,
    include_rewards: bool = True,
    include_timesteps: bool = True,
    log_scale: bool = False,
    softmax: bool = False,
) -> None:
    """
    Plot metrics from multiple experiments.

    Parameters
    ----------
    results : List[Tuple[list, list]]
        List of tuples, each containing reward_ep and timesteps_ep.
    hyperparameters_list : List[dict], optional
        List of dictionaries, each containing a set of hyperparameters.
    include_rewards : bool, optional
        Flag to include reward metrics in the plot.
    include_timesteps : bool, optional
        Flag to include timestep metrics in the plot.
    log_scale : bool, optional
        Flag to set the y-axis to a logarithmic scale.
    softmax : bool, optional
        Flag to indicate that the softmax policy is used (and the tau is to be
        included in the labels). If False, the e-greedy policy is assumed (and
        epsilon is included in the labels).

    Returns
    -------
    None
    """

    sns.set_theme(style="darkgrid")
    color_palette = sns.color_palette("husl", len(results))

    # Figure Size
    plt.figure(figsize=(16, 8))

    
    for i, (reward_ep, timesteps_ep) in enumerate(results):
        color = color_palette[i]
        lighter_color = tuple(list(color[:3]) + [0.5])  # Same RGB but with different Alpha

        hyperparameters = hyperparameters_list[i] if hyperparameters_list else {}
        episode_number = np.linspace(1, len(timesteps_ep) + 1, len(timesteps_ep) + 1)

        if softmax:
            label_suffix = f" (alpha={hyperparameters.get('alpha', 'NA')}, gamma={hyperparameters.get('gamma', 'NA')}, tau={hyperparameters.get('tau', 'NA')})"
        else: # we assume that e-greedy is used
            label_suffix = f" (alpha={hyperparameters.get('alpha', 'NA')}, gamma={hyperparameters.get('gamma', 'NA')}, epsilon={hyperparameters.get('epsilon', 'NA')})"
        
        if include_rewards:
            cum_rewards = np.cumsum(np.array(reward_ep))
            reward_per_episode_smooth = [cum_rewards[j] / episode_number[j] for j in range(len(cum_rewards))]
            plt.plot(reward_per_episode_smooth, label='Reward Smooth' + label_suffix, color=color)
            plt.plot(reward_ep, label='Reward Original' + label_suffix, color=lighter_color)

        if include_timesteps:
            cum_steps = np.cumsum(np.array(timesteps_ep))
            steps_per_episode_smooth = [cum_steps[j] / episode_number[j] for j in range(len(cum_steps))]
            plt.plot(steps_per_episode_smooth, label='Timesteps Smooth' + label_suffix, color=color)
            plt.plot(timesteps_ep, label='Timesteps Original' + label_suffix, color=lighter_color)

    # Customizing y-ticks for linear or log scale
    if log_scale:
        plt.yscale("log")
        
        # Adapt ticks for log scale
        y_min, y_max = plt.ylim()
        ticks = np.logspace(np.log10(y_min), np.log10(y_max), num=10)
        plt.yticks(ticks, [f"{tick:.2e}" for tick in ticks])

    else:
        # Customizing y-ticks for linear scale
        y_ticks = np.arange(
            int(plt.ylim()[0]), 
            int(plt.ylim()[1]) + 1, 
            step=(plt.ylim()[1] - plt.ylim()[0]) / 20
        )
        plt.yticks(y_ticks)
    
    # Labels and titles
    plt.xlabel("Episode")
    if include_rewards and include_timesteps:
        plt.ylabel("Metric Value")
        plt.title("Metrics per Episode")
    elif include_rewards:
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
    elif include_timesteps:
        plt.ylabel("Timesteps")
        plt.title("Timesteps per Episode")
    
    plt.legend()

    # saving plot
    if include_rewards and include_timesteps:
        plt.savefig("metrics_per_episode.png")
    elif include_rewards:
        plt.savefig("reward_per_episode.png")
    elif include_timesteps:
        plt.savefig("timesteps_per_episode.png")

    plt.show()



# ----------------------------------------------------------------------------
# ----------------------------    Main    ------------------------------------
# ----------------------------------------------------------------------------


# se crea el diccionario que contendrá los valores de Q
# para cada tupla (estado, acción)
q = {}


# --------------------------   Hyperparameters   -----------------------------
# # Basic starting parameters (given by the professors)
# hyperparameters = {
#     "alpha": 0.5,
#     "gamma": 1,
#     "epsilon": 0.1,
#     "tau": 25,
# }

# # Parameters for favouring exploration (still SARSA)
# hyperparameters = {
#     "alpha": 0.1,
#     "gamma": 1.5,
#     "epsilon": 0.3,
#     "tau": 25,
# }

# Parameters for favouring convergence (still SARSA, but with softmax policy)
# hyperparameters = {
#     "alpha": 0.3,
#     "gamma": 1,
#     "epsilon": 0.1,
#     "tau": 5,
# }

# -----------------------  Other choices  ------------------------------------


# Policy options: choose_action_e_greedy, choose_action_softmax
choose_action_with_policy = choose_action_softmax

# Learning algorithm options: learn_Q_learning, learn_SARSA
learning_function = learn_SARSA

# cantidad de episodios a ejecutar
episodes_to_run = 500

env = gym.make("CliffWalking-v0", render_mode="rgb_array")
actions = range(env.action_space.n)

# se declara una semilla aleatoria
random_state = np.random.RandomState(42)


# ---------------------   Execution Zone   -----------------------------------

# -- Using the functions for plotting the metrics of a single run ------------
# agent execution
# avg_rew_per_episode, timesteps_ep, reward_ep = run(
#     learning_function,
#     hyperparameters,
#     episodes_to_run,
#     env,
#     actions,
#     q,
#     random_state,
#     render=False,
# )

# plot_steps_per_episode(timesteps_ep)
# plot_steps_per_episode_smooth(timesteps_ep)
# draw_value_matrix(q)

# plot_combined_metrics(reward_ep, timesteps_ep, hyperparameters, include_rewards=False, include_timesteps=True)




# -- Using the functions for comparing convergence for a single pair of curves
# results = [(reward_ep, timesteps_ep)]
# hyperparameters_list = [hyperparameters]
# plot_multiple_metrics(results, hyperparameters_list, include_rewards=False, include_timesteps=True, log_scale=False)



# ---------- Comparing convergence for different hyperparameters -------------

# We take the hyperparameters given by the professors as base point from which
# we will change one parameter at a time. These were:
# hyperparameters = {"alpha": 0.5, "gamma": 1, "epsilon": 0.1, "tau": 25}

# changing alpha
# hyperparameters_list = [
#     {'alpha': 0.01, 'gamma': 1, 'epsilon': 0.1, 'tau': 25},
#     {'alpha': 0.1, 'gamma': 1, 'epsilon': 0.1, 'tau': 25},
#     {'alpha': 0.3, 'gamma': 1, 'epsilon': 0.1, 'tau': 25},
#     {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.1, 'tau': 25},
#     ]

# changing gamma
# hyperparameters_list = [
#     {'alpha': 0.5, 'gamma': 0.5, 'epsilon': 0.1, 'tau': 25},
#     {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.1, 'tau': 25},
#     {'alpha': 0.5, 'gamma': 1.5, 'epsilon': 0.1, 'tau': 25},
#     ]

# changing epsilon
# hyperparameters_list = [
#     {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.1, 'tau': 25},
#     {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.3, 'tau': 25},
#     {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.5, 'tau': 25},
#     ]

# changing tau
hyperparameters_list = [
    {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.1, 'tau': 5},
    {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.1, 'tau': 15},
    {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.1, 'tau': 25},
    {'alpha': 0.5, 'gamma': 1, 'epsilon': 0.1, 'tau': 50},
    ]

# --------------------- Ploting multiple curves ------------------------------


results = run_multiple_experiments(
    learning_function, 
    hyperparameters_list, 
    episodes_to_run, 
    env, 
    actions, 
    q, 
    random_state,
    # render=False
)

plot_multiple_metrics(results, hyperparameters_list, include_rewards=False, include_timesteps=True, log_scale=True, softmax=True)



env.close()
