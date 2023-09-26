'''Reinformcent Learning - Lab 1

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

'''


import itertools
from typing import Any, Callable, Tuple

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym


# -------------------- Some plotting functions -------------------------------

def plot_reward_per_episode(reward_ep) -> None:
    episode_rewards = np.array(reward_ep)

    # se suaviza la curva de convergencia
    episode_number = np.linspace(
        1, len(episode_rewards) + 1, len(episode_rewards) + 1)
    acumulated_rewards = np.cumsum(episode_rewards)

    reward_per_episode = [
        acumulated_rewards[i] / episode_number[i]
        for i in range(len(acumulated_rewards))
    ]

    plt.plot(reward_per_episode)
    plt.title("Recompensa acumulada por episodio")
    # plt.show()
    # we save the plot
    plt.savefig('reward_per_episode.png', dpi=1000)


def plot_steps_per_episode(timesteps_ep) -> None:
    # se muestra la curva de aprendizaje de los pasos por episodio
    episode_steps = np.array(timesteps_ep)
    plt.plot(np.array(range(0, len(episode_steps))), episode_steps)
    plt.title("Pasos (timesteps) por episodio")
    # plt.show()
    # we save the plot
    plt.savefig('steps_per_episode.png', dpi=1000)


def plot_steps_per_episode_smooth(timesteps_ep) -> None:
    episode_steps = np.array(timesteps_ep)

    # se suaviza la curva de aprendizaje
    episode_number = np.linspace(
        1, len(episode_steps) + 1, len(episode_steps) + 1)
    acumulated_steps = np.cumsum(episode_steps)

    steps_per_episode = [
        acumulated_steps[i] / episode_number[i] for i in range(
            len(acumulated_steps))
    ]

    plt.plot(steps_per_episode)
    plt.title("Pasos (timesteps) acumulados por episodio")
    # plt.show()
    # we save the plot
    plt.savefig('steps_per_episode_smooth.png', dpi=1000)


def draw_value_matrix(q) -> None:
    n_rows = 4
    n_columns = 12
    n_actions = 4

    # se procede con los cálculos previos a la graficación de la matriz de valor
    q_value_matrix = np.empty((n_rows, n_columns))
    for row in range(n_rows):
        for column in range(n_columns):
            state_values = []

            for action in range(n_actions):
                state_values.append(
                    q.get((row * n_columns + column, action), -100))

            maximum_value = max(
                state_values
            )  # determinamos la acción que arroja máximo valor

            q_value_matrix[row, column] = maximum_value

    # el valor del estado objetivo se asigna en -1 (reward recibido al llegar)
    # para que se coloree de forma apropiada
    q_value_matrix[3, 11] = -1

    # se grafica la matriz de valor
    plt.imshow(q_value_matrix, cmap=plt.cm.RdYlGn)
    plt.tight_layout()
    plt.colorbar()

    for row, column in itertools.product(
        range(q_value_matrix.shape[0]), range(q_value_matrix.shape[1])
    ):
        left_action = q.get((row * n_columns + column, 3), -1000)
        down_action = q.get((row * n_columns + column, 2), -1000)
        right_action = q.get((row * n_columns + column, 1), -1000)
        up_action = q.get((row * n_columns + column, 0), -1000)

        arrow_direction = "D"
        best_action = down_action

        if best_action < right_action:
            arrow_direction = "R"
            best_action = right_action
        if best_action < left_action:
            arrow_direction = "L"
            best_action = left_action
        if best_action < up_action:
            arrow_direction = "U"
            best_action = up_action
        if best_action == -1:
            arrow_direction = ""

        # notar que column, row están invertidos en orden en la línea de abajo
        # porque representan a x,y del plot
        plt.text(column, row, arrow_direction, horizontalalignment="center")

    plt.xticks([])
    plt.yticks([])
    # plt.show()
    # we save the plot
    plt.savefig('value_matrix.png')

    print("\n Matriz de mejor acción-valor (en números): \n\n", q_value_matrix)


# ------------------------ Policies ------------------------------------------

def choose_action_e_greedy(
        state: int,
        actions: range,
        q: dict,
        hyperparameters: dict,
        random_state: np.random.RandomState,
) -> int:
    """
    Elije una acción de acuerdo al aprendizaje realizado previamente
    usando una política de exploración épsilon-greedy
    """
    # ej: para 4 acciones inicializa en [0,0,0,0]
    q_values = [q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    # sorteamos un número: es menor a épsilon?
    if random_state.uniform() < hyperparameters['epsilon']:
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
    """
    Choose an action according to the learning done previously
    using a softmax policy.
    """
    # we get the value of each action in this state
    q_values = [q.get((state, a), 0.0) for a in actions]
    
    # we get the temperature to be used
    tau = hyperparameters['tau']
    
    # we calculate the probabilities of the policy
    policy = [np.exp(q_values[a]/tau) for a in actions]
    norm = sum(policy)
    policy = [p/norm for p in policy]

    # now, we select action i with probability policy[i]
    # (we get a random number and then compare it with
    # the cumulative sum of the probabilities)
    rand_value = random_state.uniform()
    i = 0
    while rand_value > 0:
        rand_value -= policy[i]
        i += 1
    # note that i is incremented one more time than needed

    return actions[i-1]


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
    """Update the Q-value for the given state and action pair
    using SARSA learning (on-policy TD control).

    Args:
        state (int): the current state
        action (int): the current action
        reward (int): the reward obtained by taking action `action` in state `state`
        next_state (int): the next state
        next_action (int): the next action
    """

    # initialize the Q-value for brand-new state-action pairs
    if (state, action) not in q:
        q[(state, action)] = 0.0  # or another initial value
    if (next_state, next_action) not in q:
        q[(next_state, next_action)] = 0.0  # or another initial value
    
    # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
    q[(state, action)] += hyperparameters['alpha'] * (
        reward + hyperparameters['gamma'] * q[(next_state, next_action)] -
        q[(state, action)])


def learn_Q_learning(
        state: int,
        action: int,
        reward: int,
        next_state: int,
        next_action: int,
        q: dict,
        hyperparameters: dict,
) -> None:
    """Update the Q-value for the given state and action pair
    using Q-learning algorithm (off-policy TD control).

    Args:
        state (int): the current state
        action (int): the current action
        reward (int): the reward obtained by taking action `action` in state `state`
        next_state (int): the next state
        next_action (int): the next action

    Notes
    -----
    I keep the next_action parameter for consistency with the SARSA function.
    In this way, I can use the both with the same signature.
    """

    # initialize the Q-value for brand-new state-action pairs
    if (state, action) not in q:
        q[(state, action)] = 0.0  # or another initial value
    if (next_state, next_action) not in q:
        q[(next_state, next_action)] = 0.0  # or another initial value
    
    # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
    q_max = max([q.get((next_state, a), 0.0) for a in range(4)])
    q[(state, action)] += hyperparameters['alpha'] * (
        reward + hyperparameters['gamma'] * q_max -
        q[(state, action)])


# ------------------ Training Loop ------------------------

def run(
    learning_function: Callable,
    hyperparameters: dict,
    episodes_to_run: int,
    env: gym.Env,
    actions: range,
    q: dict,
    random_state: np.random.RandomState,
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

    for _ in range(episodes_to_run):
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
            state, actions, q, hyperparameters, random_state)

        while not done:
            # el agente ejecuta la acción elegida y obtiene los resultados
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_action = choose_action_with_policy(
                next_state, actions, q, hyperparameters, random_state)

            episode_reward += reward
            learning_function(state, action, reward, next_state, next_action, q, hyperparameters)

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

    return reward_of_episode.mean(), timesteps_of_episode, reward_of_episode


# ------------------ Main ------------------------

# se crea el diccionario que contendrá los valores de Q
# para cada tupla (estado, acción)
q = {}

# definimos sus híper-parámetros básicos
hyperparameters = {
    "alpha": 0.5,
    "gamma": 1,
    "epsilon": 0.1,
    "tau": 25,
}

# política para elegir acciones
choose_action_with_policy = choose_action_e_greedy

# se puede cambiar por learn_Q_learning, una vez que se implemente
learning_function = learn_SARSA

# cantidad de episodios a ejecutar
episodes_to_run = 500

env = gym.make("CliffWalking-v0")
actions = range(env.action_space.n)

# se declara una semilla aleatoria
random_state = np.random.RandomState(42)

# agent execution
avg_rew_per_episode, timesteps_ep, reward_ep = run(
    learning_function,
    hyperparameters,
    episodes_to_run,
    env,
    actions,
    q,
    random_state
)

plot_steps_per_episode(timesteps_ep)
plot_steps_per_episode_smooth(timesteps_ep)
draw_value_matrix(q)

env.close()