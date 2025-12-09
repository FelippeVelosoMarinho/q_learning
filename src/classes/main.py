import numpy as np
import matplotlib.pyplot as plt

alpha = 0.2 
y = 0.95 
limit = 10

actions = ["CIMA", "DIREITA", "BAIXO", "ESQUERDA"]

# o estado de inicializa¸c˜ao do agente em cada epis´odio deve ser escolhido aleat´orimente
#  dentre os estados n˜ao terminais.

# PRECISA ALTERAR TODA POLÍTICA DE ATUALIZAÇÃO PARA A DO ENUNCIADO

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

def e_greedy_action(state: tuple[int, int], qtable: np.ndarray, epsilon: float) -> int:
    state_idx = k2pos(state)
    
    # Ações válidas (que não são -inf)
    valid_actions = np.where(qtable[state_idx] != -np.inf)[0]
    
    if np.random.rand() < epsilon:
        # Exploração: aleatório entre as válidas
        return np.random.choice(valid_actions)
    else:
        # Explotação: melhor ação com desempate aleatório
        q_values = qtable[state_idx, valid_actions]
        max_q = np.max(q_values)
        # Pega todos os índices que têm o valor máximo
        best_actions_indices = np.where(qtable[state_idx] == max_q)[0]
        # Filtra para garantir que só pegamos ações válidas (precaução)
        best_valid_actions = [a for a in best_actions_indices if a in valid_actions]
        
        return np.random.choice(best_valid_actions)

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

def qlearning(verbose=False, interactive=False, total_episodes=200, epsilon=0.1):
    qtable = get_table()
    reward_history = []
    terminal_states = [(4,4), (4,2), (1,3)]
    
    for i in range(total_episodes):
        step = 0
        cummulated_reward = 0
        
        # Sorteio de estado inicial válido
        while True:
            start_x = np.random.randint(1,5)
            start_y = np.random.randint(1,5)
            s = [start_x, start_y]
            if tuple(s) not in terminal_states:
                break

        while step < limit:
            action = e_greedy_action(s, qtable, epsilon)
            s_next = rollout(s, action)
            reward = get_reward(s_next)
            
            cummulated_reward += reward
            
            # atualização da Q-Table
            q_current = qtable[k2pos(s), action]
            
            # Verifica se o próximo estado é terminal para definir o alvo
            if reward in [20, -20]:
                target = reward
            else:
                q_next_max = np.max(qtable[k2pos(s_next)])
                target = reward + y * q_next_max
            
            # Fórmula corrigida (subtraindo q_current)
            qtable[k2pos(s), action] = q_current + alpha * (target - q_current)
            # =========================================

            s = s_next
            
            if reward in [20, -20]:
                break
            step += 1
            
        reward_history.append(cummulated_reward)
        
    return qtable, reward_history

if __name__ == "__main__":
    print(get_table())
    table, history = qlearning()
    #x = np.linspace(1, 100, 100)
    x = range(len(history))
    #plt.plot(x, history)
    #plt.show()
    print(table)
