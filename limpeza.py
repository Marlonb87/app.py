import pandas as pd
import re

# Caminho do arquivo original
arquivo = "carteira de fabricação 21.08.2025rev2.xlsx"

# Ler a aba Relatorio sem considerar cabeçalho
df = pd.read_excel(arquivo, sheet_name="Relatorio", header=None)

# Função para limpar valores
def limpar_valor(valor):
    if pd.isna(valor):
        return ""  # Retorna vazio se nulo
    valor_str = str(valor).strip()  # Converte para texto e remove espaços
    # Mantém apenas números e "/"
    valor_limpo = "".join(re.findall(r"[0-9/]", valor_str))
    return valor_limpo

# Aplicar limpeza nas colunas A (0) e I (8)
df[0] = df[0].apply(limpar_valor)  # Coluna A


# Salvar em novo arquivo limpo
df.to_excel("Relatorio_Limpo.xlsx", index=False, header=False)

print("Arquivo limpo criado: 'Relatorio_Limpo.xlsx'")