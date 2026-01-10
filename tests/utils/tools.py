import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rsa_depth_correlation(filtered_rsa_maps, filtered_depth_maps):
    """
    Plota um gráfico de correlação entre os valores de RSA e a profundidade dos resíduos.
    
    Args:
        filtered_rsa_maps (List[np.ndarray]): Lista de mapas RSA filtrados.
        filtered_depth_maps (List[np.ndarray]): Lista de mapas de profundidade filtrados.
    """
    # Garantir que as listas de RSA e profundidade estejam alinhadas
    rsa_values = np.concatenate(filtered_rsa_maps)
    depth_values = np.concatenate(filtered_depth_maps)

    if len(rsa_values) != len(depth_values):
        raise ValueError("Os tamanhos de RSA e profundidade não coincidem!")
    
    # Criar o scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=depth_values, y=rsa_values, alpha=0.7)
    
    # Adicionar linha de tendência
    sns.regplot(x=depth_values, y=rsa_values, scatter=False, color='red', ci=None)

    # Configurações do gráfico
    plt.title("Correlação entre Profundidade e RSA dos Resíduos", fontsize=14)
    plt.xlabel("Profundidade do Resíduo", fontsize=12)
    plt.ylabel("Valor de RSA", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Salvar o gráfico no caminho especificado
    plt.savefig("correlation_Virus.png", format="png", dpi=300)
    plt.close()  # Fecha a figura para liberar memória

    print(f"Gráfico salvo")