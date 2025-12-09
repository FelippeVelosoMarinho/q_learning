import numpy as np
import sys

# Importa as funções do seu arquivo main.py
sys.path.insert(1, './src/classes') 
from main import qlearning

def gerar_latex():
    # 1. Executa o treinamento para obter a tabela preenchida
    # Usamos epsilon=0.01 pois foi o que deu melhor resultado na sua análise
    print("Treinando o agente para gerar os dados...")
    qtable, _ = qlearning(total_episodes=300, epsilon=0.01)

    # Configuração dos cabeçalhos
    headers = ["Estado", "CIMA", "DIREITA", "BAIXO", "ESQUERDA"]
    
    print("\n" + "="*50)
    print("COPIE O CÓDIGO ABAIXO PARA O SEU RELATÓRIO:")
    print("="*50 + "\n")

    # Início do ambiente de tabela LaTeX
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize") # Fonte menor para caber na página
    print(r"\begin{tabular}{|c|r|r|r|r|}")
    print(r"\hline")
    
    # Cabeçalho da tabela com negrito
    header_str = " & ".join([f"\\textbf{{{h}}}" for h in headers]) + r" \\ \hline"
    print(header_str)

    # 2. Itera sobre as linhas da Q-Table (0 a 15)
    # A ordem 0->15 segue a lógica (1,1), (1,2)... até (4,4) conforme seu k2pos
    for idx in range(16):
        # Converte índice linear de volta para coordenadas (Row, Col)
        # Sua fórmula k2pos é: (row-1)*4 + (col-1)
        row = (idx // 4) + 1
        col = (idx % 4) + 1
        state_label = f"({row}, {col})"

        row_values = []
        for action in range(4):
            val = qtable[idx, action]
            
            # Tratamento para -inf (bordas)
            if val == -np.inf:
                row_values.append(r"$-\infty$")
            else:
                # Formata para 2 casas decimais
                row_values.append(f"{val:.2f}")

        # Monta a linha da tabela
        line = f"{state_label} & " + " & ".join(row_values) + r" \\ \hline"
        print(line)

    # Fechamento da tabela
    print(r"\end{tabular}")
    print(r"\caption{Valores Q finais aprendidos (Média de 300 episódios, $\epsilon=0.01$)}")
    print(r"\label{tab:qtable_numeric}")
    print(r"\end{table}")

if __name__ == "__main__":
    gerar_latex()