import numpy as np
import sys
import os

# Importa sua função de qlearning
# Certifique-se de que o caminho está correto para o seu projeto
sys.path.insert(1, './src/classes') 
from main import qlearning

# Mapeamento das ações (índice -> nome)
# 0: CIMA, 1: DIREITA, 2: BAIXO, 3: ESQUERDA
actions_map = {0: "CIMA", 1: "DIREITA", 2: "BAIXO", 3: "ESQUERDA"}
arrows_map = {0: r"$\uparrow$ CIMA", 1: r"$\rightarrow$ DIREITA", 2: r"$\downarrow$ BAIXO", 3: r"$\leftarrow$ ESQUERDA"}

def get_state_index(row, col):
    # AJUSTE AQUI se sua lógica de índice for diferente.
    # A maioria das implementações usa: (row - 1) * 4 + (col - 1)
    # Considerando grid 4x4 com índices de 1 a 4.
    return (row - 1) * 4 + (col - 1)

def print_results():
    # 1. Treina o modelo com o melhor Epsilon encontrado
    print("Treinando agente com Epsilon = 0.01...")
    q_table, _ = qlearning(total_episodes=300, epsilon=0.01)

    print("\n" + "="*60)
    print(f"{'Posição':<10} | {'Melhor Ação':<20} | {'Valor Q':<10}")
    print("="*60)

    # Definição dos estados especiais para exibir no texto
    special_states = {
        (4, 2): "TÓXICO (-20)",
        (1, 3): "TÓXICO (-20)",
        (3, 2): "LAMA (-5)",
        (2, 4): "LAMA (-5)",
        (4, 4): "SAÍDA (+20)"
    }

    # 2. Itera sobre o Grid (Linha 4 até 1, Coluna 1 até 4)
    # Isso garante a ordem visual correta (topo para base)
    latex_rows = []
    
    for row in [4, 3, 2, 1]:
        row_str = []
        for col in [1, 2, 3, 4]:
            
            # Obtém o índice na Q-Table
            # Nota: Se sua Q-Table for um dicionário (row,col), altere para: q_values = q_table[(row, col)]
            try:
                state_idx = get_state_index(row, col)
                q_values = q_table[state_idx] # Assume q_table é numpy array [16, 4]
            except:
                # Fallback se for dicionário ou outra estrutura
                q_values = [0, 0, 0, 0] 

            # Encontra a melhor ação (argmax)
            best_action_idx = np.argmax(q_values)
            best_action_name = actions_map[best_action_idx]
            max_q_val = q_values[best_action_idx]

            # Verifica se é estado especial
            coord = (row, col)
            if coord in special_states:
                display_text = f"** {special_states[coord]} **"
                latex_text = f"\\textbf{{{special_states[coord].split()[0]}}}" # Pega só a palavra (TÓXICO/LAMA)
            else:
                display_text = f"{best_action_name}"
                latex_text = arrows_map[best_action_idx]

            print(f"({row}, {col}){' ':<4} | {display_text:<20} | {max_q_val:.2f}")
            
            # Monta string para tabela LaTeX (formato simples)
            row_str.append(f"({row},{col}) {latex_text}")
        
        latex_rows.append(row_str)

    # 3. Gera código LaTeX pronto para copiar
    print("\n" + "="*60)
    print("CÓDIGO PARA COPIAR NO LATEX (Section 'Política Ótima'):")
    print("="*60)
    
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    
    for r_idx, r_data in enumerate(latex_rows):
        # Junta as colunas com ' & ' e adiciona quebra de linha
        line = " & ".join(r_data) + " \\\\ \\hline"
        print(line)
        
    print("\\end{tabular}")
    print("\\caption{Política Ótima obtida com $\\epsilon=0.01$}")
    print("\\label{tab:policy_final}")
    print("\\end{table}")

if __name__ == "__main__":
    print_results()