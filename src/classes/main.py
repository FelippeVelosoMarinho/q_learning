import numpy as np
import matplotlib.pyplot as plt

alpha = 0.2 
y = 0.95 
epsilon = 0.1 # ϵ-greedy param
limit = 10

actions = ["CIMA", "DIREITA", "BAIXO", "ESQUERDA"]

# o estado de inicializa¸c˜ao do agente em cada epis´odio deve ser escolhido aleat´orimente
#  dentre os estados n˜ao terminais.

# PRECISA ALTERAR TODA POLÍTICA DE ATUALIZAÇÃO PARA A DO ENUNCIADO

episodes = [[2, 2],
            [3, 1],
            [1, 1],
            [1, 2],
            [1, 4],
            [1, 4],
            [2, 2],
            [3, 1],
            [1, 1]]

# state vai ser a posicao (x, y) no grid 4x4

def rollout(state, action):
    '''
    Dinamica antiga do ambiente (incrementa que nem no Ed3)
    PRECISA ALTERAR PRA POLítica do enunciado
    
    :param state: Description
    :param action: Description
    '''
    new_state = state[:]
    if action == 0:
        new_state[0] += 1
    if action == 1:
        new_state[1] += 1
    if action == 2:
        new_state[0] -= 1
    if action == 3:
        new_state[1] -= 1

    return new_state

def get_reward(state: tuple[int, int]) -> int:
    state = tuple(state)
    if state in [(4, 2), (1, 3)]:
        return -20
    if state in [(3, 2), (2, 4)]:
        return -5
    if state == (4, 4):
        return 20
    return -1

def get_table():
    '''
    Inicializa uma matriz de zeros com dimensão (16,4)
    Os laços verificam as bordas do grid. Caso uma ação leve o agente fora
    do grid, o valor dessa célula da tabela recebe -np.inf
    '''
    table = np.zeros((16, 4))
    for i in range(1, 5):
        for j in range(1, 5):
            if i == 1:
                table[k2pos((i, j)), 2] = -np.inf
            if j == 1:
                table[k2pos((i, j)), 3] = -np.inf
            if i == 4:
                table[k2pos((i, j)), 0] = -np.inf
            if j == 4:
                table[k2pos((i, j)), 1] = -np.inf
    return table

def k2pos(state: tuple[int, int]) -> int:
    '''
    Auxiliar para achatamento de coordenadas. quando um agente (x,y) 
    precisa ser convertido para um índice de 0 a 15 (linhas da tabela)
    Assim normalizamos os estados de 1 a 4 para 0 a 3.
    
    :param state: Estado atual
    :type state: tuple[int, int]
    :return: inteiro normalizado de 0 a 3
    :rtype: int
    '''
    return (state[0]-1)*4 + (state[1] - 1)

def qlearning(verbose=False, interactive=False, total_episodes=500):
    qtable = get_table()
    reward_history = []
    terminal_states = [(4,4), (4,2), (1,3)]
    
    for i in range(total_episodes):
        step = 0
        cummulated_reward = 0
        
        while True: # loop de sorteio
            start_x = np.random.randint(1,5)
            start_y = np.random.randint(1,5)
            s = [start_x, start_y]
            
            if tuple(s) not in terminal_states:
                break

        while step < limit:
            action = np.argmax(qtable[k2pos(s), :])
            s_next = rollout(s, action)
            reward = get_reward(s_next)
            cummulated_reward += reward
            q_current = qtable[k2pos(s), action]
            q_next_max = np.max(qtable[k2pos(s_next)])
            qtable[k2pos(s), action] = q_current + alpha * (reward + y * q_next_max )

            s = s_next
            if verbose or interactive:
                print("tabela no passo ", step, "episodio ", s, "reward ", reward, "acao ", actions[action])
                if interactive:
                    input()
                print(action)
                print(qtable)

            if reward in [20, -20]:
                break
            step += 1
        if (verbose):
            print("Recompensa acumulada no episodio ", s, " : ", cummulated_reward)
        reward_history.append(cummulated_reward)
    return qtable, reward_history

if __name__ == "__main__":
    print(get_table())
    table, history = qlearning()
    x = np.linspace(1, 100, 100)
    plt.plot(x, history)
    plt.show()
    print(table)
