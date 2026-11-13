

"""Funcoes auxiliares para a otimizacao de passagens da EFVM."""

import csv
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURACOES
# ============================================================

PASTA = Path(__file__).resolve().parent

CENARIO = 3
CAPACIDADE = 2250
CLASSE = "economica"

REPETICOES = 100
SEMENTE = 42
VARIACAO = 0.20

# Ordem original: Vitoria -> Belo Horizonte.
# O programa inverte para BH -> Vitoria.
ESTACOES = [
    "Pedro Nolasco",
    "Flexal",
    "Fundão",
    "Aricanga",
    "Piraqueaçu",
    "Colatina",
    "Itapina",
    "Mascarenhas",
    "Baixo Guandu",
    "Aimorés",
    "Resplendor",
    "Crenaque",
    "Conselheiro Pena",
    "Barra do Cuieté",
    "São Tomé do Rio Doce",
    "Tumiritinga",
    "Governador Valadares",
    "Pedra Corrida",
    "Periquito",
    "Frederico Sellow",
    "Ipaba",
    "Intendente Câmara",
    "Mário Carvalho",
    "Antônio Dias",
    "Desembargador Drumond",
    "João Monlevade",
    "Rio Piracicaba",
    "Dois Irmãos",
    "Belo Horizonte",
]

# Para testar 5 estacoes:
#SELECIONADAS = [0, 5, 16, 21, 28]

#SELECIONADAS = [0, 5, 8, 10, 16, 21, 24, 25, 27, 28]

# Para testar todas:
SELECIONADAS = list(range(len(ESTACOES)))


# ============================================================
# POPULACOES DE REFERENCIA
# ============================================================

# Dados populacionais fornecidos para o projeto.
# A associacao entre estacoes e areas atendidas e uma hipotese.
#
# Municipios com mais de uma estacao:
# a populacao e dividida igualmente entre elas.
#
# Pedro Nolasco: metade de Cariacica + Vitoria,
# Vila Velha e Serra.
# Flexal: outra metade de Cariacica.
#
# Crenaque: associacao provisoria com Resplendor.

POPULACOES = {
    "Belo Horizonte": 2315560,
    "Dois Irmãos": 30778,
    "Rio Piracicaba": 14631,
    "João Monlevade": 80187,
    "Desembargador Drumond": 17438,
    "Antônio Dias": 9219,
    "Mário Carvalho": 81579 + 104736,
    "Intendente Câmara": 227731,
    "Ipaba": 17136,
    "Frederico Sellow": 23928,
    "Periquito": 6553 / 2,
    "Pedra Corrida": 6553 / 2,
    "Governador Valadares": 257171,
    "Tumiritinga": 5886 / 2,
    "São Tomé do Rio Doce": 5886 / 2,
    "Barra do Cuieté": 20824 / 2,
    "Conselheiro Pena": 20824 / 2,
    "Crenaque": 17226 / 2,
    "Resplendor": 17226 / 2,
    "Aimorés": 25269,
    "Baixo Guandu": 30674 / 2,
    "Mascarenhas": 30674 / 2,
    "Itapina": 120033 / 2,
    "Colatina": 120033 / 2,
    "Piraqueaçu": 14079,
    "Aricanga": 94765,
    "Fundão": 18014,
    "Flexal": 353491 / 2,
    "Pedro Nolasco": 353491 / 2 + 322869 + 467722 + 520653,
}


# ============================================================
# PERFIS DE DEMANDA
# ============================================================

# Cada valor corresponde a demanda de referencia do par
# origem-destino com maior produto populacional.
# NAO corresponde a demanda total do trem.

PERFIS = {
    "baixa": 1800,
    "media": 2200,
    "alta": 4000,
}


# ============================================================
# DEMANDA FIXA DO CENARIO 2
# ============================================================

DEMANDA_EXEMPLO = {
    ("Belo Horizonte", "Intendente Câmara"): 800,
    ("Belo Horizonte", "Governador Valadares"): 400,
    ("Belo Horizonte", "Colatina"): 380,
    ("Belo Horizonte", "Pedro Nolasco"): 600,
    ("Intendente Câmara", "Governador Valadares"): 650,
    ("Intendente Câmara", "Colatina"): 250,
    ("Intendente Câmara", "Pedro Nolasco"): 240,
    ("Governador Valadares", "Colatina"): 500,
    ("Governador Valadares", "Pedro Nolasco"): 320,
    ("Colatina", "Pedro Nolasco"): 450,
}


# ============================================================
# VALIDACAO
# ============================================================

def validar_configuracao():

    if CENARIO not in (1, 2, 3):
        raise ValueError("CENARIO deve ser 1, 2 ou 3.")

    if CAPACIDADE <= 0 or not isinstance(CAPACIDADE, int):
        raise ValueError("CAPACIDADE deve ser inteiro positivo.")

    if CLASSE not in ("economica", "executiva"):
        raise ValueError("Classe invalida.")

    if REPETICOES < 1:
        raise ValueError("REPETICOES deve ser positiva.")

    if not 0 <= VARIACAO <= 1:
        raise ValueError("VARIACAO deve estar entre 0 e 1.")

    if (
        len(SELECIONADAS) < 2
        or SELECIONADAS != sorted(set(SELECIONADAS))
        or any(i < 0 or i >= len(ESTACOES) for i in SELECIONADAS)
    ):
        raise ValueError("Indices das estacoes invalidos.")

    if set(POPULACOES) != set(ESTACOES):
        raise ValueError("Populacoes incompletas ou estacoes desconhecidas.")


# ============================================================
# LEITURA DAS TARIFAS
# ============================================================

def carregar_tarifas(arquivo, mercados):

    tabela = {}

    with arquivo.open(
        encoding="utf-8-sig", newline=""
    ) as f:

        leitor = csv.DictReader(f)

        for linha in leitor:

            origem = linha["origem"].strip()
            destino = linha["destino"].strip()

            valor = float(
                linha[CLASSE].replace(",", ".")
            )

            tabela[origem, destino] = valor

    tarifas = {}

    for origem, destino in mercados:

        if (origem, destino) in tabela:
            tarifas[origem, destino] = tabela[origem, destino]

        elif (destino, origem) in tabela:
            tarifas[origem, destino] = tabela[destino, origem]

        else:
            raise ValueError(
                f"Tarifa nao encontrada: {origem} -> {destino}"
            )

    return tarifas


# ============================================================
# DEMANDA FIXA - CENARIO 2
# ============================================================

def carregar_demanda_fixa(mercados):

    arquivo = PASTA / "demandas_cenario2_BH_Vitoria.csv"

    if not arquivo.exists():

        with arquivo.open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:

            writer = csv.writer(f)
            writer.writerow(["origem", "destino", "demanda"])

            for origem, destino in mercados:
                valor = DEMANDA_EXEMPLO.get(
                    (origem, destino), ""
                )
                writer.writerow([origem, destino, valor])

        print(f"Arquivo criado: {arquivo.name}")

    dados = {}

    with arquivo.open(
        encoding="utf-8-sig", newline=""
    ) as f:

        for linha in csv.DictReader(f):

            origem = linha["origem"].strip()
            destino = linha["destino"].strip()
            valor = linha["demanda"].strip()

            if not valor:
                continue

            numero = int(valor)

            if numero < 0:
                raise ValueError("Demanda negativa.")

            if (origem, destino) in dados:
                raise ValueError(
                    f"Demanda duplicada: {origem} -> {destino}"
                )

            dados[origem, destino] = numero

    faltantes = [
        od for od in mercados if od not in dados
    ]

    if faltantes:
        raise ValueError(
            f"Preencha as demandas faltantes no CSV: {faltantes}"
        )

    return {od: dados[od] for od in mercados}


# ============================================================
# DEMANDA POPULACIONAL - CENARIO 3
# ============================================================

def calcular_pesos_populacionais(mercados):
    """
    Calcula o produto das populacoes de origem e destino.
    A distancia nao e considerada.
    """

    pesos = {}

    for origem, destino in mercados:

        pesos[origem, destino] = (
            POPULACOES[origem] * POPULACOES[destino]
        )

    return pesos


def gerar_demanda(mercados, perfil, pesos, rng):
    """
    Gera demanda inteira para cada origem-destino.

    D_ref = D_max * peso / maior_peso

    A demanda sorteada varia +/-20% em torno da referencia.
    """

    demanda_maxima = PERFIS[perfil]
    maior_peso = max(pesos.values())

    demandas = {}

    for od in mercados:

        demanda_referencia = (
            demanda_maxima * pesos[od] / maior_peso
        )

        fator_aleatorio = rng.uniform(
            1 - VARIACAO,
            1 + VARIACAO
        )

        demanda = round(
            demanda_referencia * fator_aleatorio
        )

        demandas[od] = max(0, demanda)

    return demandas


# ============================================================
# APRESENTACAO DOS RESULTADOS
# ============================================================

def imprimir_resultado(
    cidades, mercados, tarifas, demanda, resultado
):

    print(
        f"\nReceita maxima: R$ {resultado['receita']:,.2f}"
    )
    print(
        f"Tempo de resolucao: {resultado['tempo']:.6f} s"
    )
    print(
        f"Passagens vendidas: {resultado['passagens']}"
    )

    print("\nOrigem -> Destino | Vendidas | Demanda | Tarifa")

    for origem, destino in mercados:

        vendidas = resultado["vendas"][origem, destino]

        if demanda is None:
            d = "ilimitada"
        else:
            d = demanda[origem, destino]

        print(
            f"{origem} -> {destino} | "
            f"{vendidas} | {d} | "
            f"R$ {tarifas[origem, destino]:.2f}"
        )

    print("\nOcupacao por trecho:")

    for s, ocupacao in enumerate(resultado["ocupacoes"]):

        percentual = 100 * ocupacao / CAPACIDADE

        print(
            f"{cidades[s]} -> {cidades[s+1]}: "
            f"{ocupacao}/{CAPACIDADE} "
            f"({percentual:.1f}%)"
        )

    if demanda is not None:

        total = sum(demanda.values())

        if total > 0:
            percentual = (
                100 * resultado["passagens"] / total
            )

            print(
                f"\nAtendimento da demanda: "
                f"{resultado['passagens']}/{total} "
                f"({percentual:.1f}%)"
            )
        else:
            print("\nDemanda total: zero")
