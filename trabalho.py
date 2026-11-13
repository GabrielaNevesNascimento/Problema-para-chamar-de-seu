
"""Otimizacao de passagens da EFVM por Programacao Linear Inteira."""

import csv
import itertools
import statistics
import time

import numpy as np
import pulp

from funcoesauxiliares import (
    PASTA,
    CAPACIDADE,
    CENARIO,
    CLASSE,
    REPETICOES,
    SEMENTE,
    ESTACOES,
    SELECIONADAS,
    PERFIS,
    validar_configuracao,
    carregar_tarifas,
    carregar_demanda_fixa,
    calcular_pesos_populacionais,
    gerar_demanda,
    imprimir_resultado,
)


# ============================================================
# FORMULACAO E RESOLUCAO DO PLI
# ============================================================

def resolver_modelo(cidades, mercados, tarifas, demanda=None):

    n = len(cidades)

    modelo = pulp.LpProblem(
        "Alocacao_EFVM",
        pulp.LpMaximize
    )

    # --------------------------------------------------------
    # 1. VARIAVEIS DE DECISAO
    # --------------------------------------------------------

    # x[i,j] = quantidade de passagens vendidas de i para j

    x = {
        (i, j): pulp.LpVariable(
            f"x_{i}_{j}",
            lowBound=0,
            cat=pulp.LpInteger
        )
        for i in range(n)
        for j in range(i + 1, n)
    }

    # --------------------------------------------------------
    # 2. FUNCAO OBJETIVO
    # --------------------------------------------------------

    # Maximizar a receita total com a venda de passagens.

    modelo += pulp.lpSum(
        tarifas[cidades[i], cidades[j]] * x[i, j]
        for i, j in x
    )

    # --------------------------------------------------------
    # 3. RESTRICOES DE CAPACIDADE
    # --------------------------------------------------------

    # A ocupacao em cada trecho nao pode ultrapassar
    # a capacidade do trem.

    for s in range(n - 1):

        modelo += (
            pulp.lpSum(
                x[i, j]
                for i, j in x
                if i <= s < j
            ) <= CAPACIDADE,
            f"Capacidade_{s}"
        )

    # --------------------------------------------------------
    # 4. RESTRICOES DE DEMANDA
    # --------------------------------------------------------

    # Cenario 1: sem limite de demanda.
    # Cenarios 2 e 3: vendas limitadas pela demanda O-D.

    if demanda is not None:

        for i, j in x:

            modelo += (
                x[i, j] <= demanda[cidades[i], cidades[j]],
                f"Demanda_{i}_{j}"
            )

    # --------------------------------------------------------
    # 5. RESOLUCAO
    # --------------------------------------------------------

    inicio = time.perf_counter()

    status = modelo.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )

    tempo = time.perf_counter() - inicio

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(
            f"Solver: {pulp.LpStatus[status]}"
        )

    # --------------------------------------------------------
    # 6. RESULTADOS
    # --------------------------------------------------------

    vendas = {
        (cidades[i], cidades[j]): int(
            round(pulp.value(x[i, j]))
        )
        for i, j in x
    }

    receita = sum(
        tarifas[od] * vendas[od]
        for od in mercados
    )

    passagens = sum(vendas.values())

    ocupacoes = []

    for s in range(n - 1):

        ocupacao = sum(
            vendas[cidades[i], cidades[j]]
            for i, j in x
            if i <= s < j
        )

        ocupacoes.append(ocupacao)

    ocupacao_media = (
        100 * statistics.mean(ocupacoes) / CAPACIDADE
    )

    atendimento = None

    if demanda is not None:

        total_demanda = sum(demanda.values())

        if total_demanda > 0:
            atendimento = (
                100 * passagens / total_demanda
            )

    return {
        "receita": receita,
        "passagens": passagens,
        "vendas": vendas,
        "ocupacoes": ocupacoes,
        "ocupacao_media_pct": ocupacao_media,
        "atendimento_pct": atendimento,
        "tempo": tempo,
    }


# ============================================================
# CENARIO 3 - SIMULACOES
# ============================================================

def simular_cenario_3(cidades, mercados, tarifas):

    rng = np.random.default_rng(SEMENTE)

    pesos = calcular_pesos_populacionais(mercados)

    registros = []

    por_perfil = {
        perfil: [] for perfil in PERFIS
    }

    print(
        f"\nCenario 3 | {len(cidades)} estacoes | "
        f"{len(mercados)} variaveis | "
        f"{REPETICOES} repeticoes por perfil"
    )

    for repeticao in range(1, REPETICOES + 1):

        # Cada repeticao gera demandas diferentes.
        # O mesmo conjunto de fatores aleatorios e usado
        # nos tres perfis, facilitando a comparacao.
        fatores = rng.uniform(
            0, 1, size=len(mercados)
        )

        for perfil in PERFIS:

            # Reinicializa um gerador com os mesmos fatores
            # para os tres perfis desta repeticao.
            class GeradorPareado:
                def __init__(self, valores):
                    self.valores = iter(valores)

                def uniform(self, minimo, maximo):
                    u = next(self.valores)
                    return minimo + (maximo - minimo) * u

            gerador = GeradorPareado(fatores)

            demanda = gerar_demanda(
                mercados, perfil, pesos, gerador
            )

            resultado = resolver_modelo(
                cidades, mercados, tarifas, demanda
            )

            registro = {
                "perfil": perfil,
                "repeticao": repeticao,
                "receita": resultado["receita"],
                "passagens": resultado["passagens"],
                "demanda_total": sum(demanda.values()),
                "ocupacao_media_pct": resultado[
                    "ocupacao_media_pct"
                ],
                "atendimento_pct": resultado[
                    "atendimento_pct"
                ],
                "tempo_s": resultado["tempo"],
            }

            registros.append(registro)
            por_perfil[perfil].append(registro)

    # --------------------------------------------------------
    # RESUMO POR PERFIL
    # --------------------------------------------------------

    for perfil, resultados in por_perfil.items():

        def media(campo):
            return statistics.mean(
                r[campo] for r in resultados
            )

        receitas = [
            r["receita"] for r in resultados
        ]

        print(f"\nPerfil: {perfil}")

        print(
            f"Receita media: R$ {media('receita'):,.2f}"
        )

        if len(receitas) > 1:
            desvio = statistics.stdev(receitas)
            print(
                f"Desvio padrao da receita: R$ {desvio:,.2f}"
            )

        print(
            f"Demanda total media: "
            f"{media('demanda_total'):.1f}"
        )

        print(
            f"Passagens medias: {media('passagens'):.1f}"
        )

        print(
            f"Ocupacao media: "
            f"{media('ocupacao_media_pct'):.1f}%"
        )

        atendimentos = [
            r["atendimento_pct"]
            for r in resultados
            if r["atendimento_pct"] is not None
        ]

        if atendimentos:
            print(
                f"Atendimento medio: "
                f"{statistics.mean(atendimentos):.1f}%"
            )

        print(
            f"Tempo mediano: "
            f"{statistics.median(r['tempo_s'] for r in resultados):.4f} s"
        )

    # --------------------------------------------------------
    # EXPORTACAO PARA CSV
    # --------------------------------------------------------

    arquivo = PASTA / "resultados_cenario3.csv"

    with arquivo.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(registros[0])
        )

        writer.writeheader()
        writer.writerows(registros)

    print(
        f"\nResultados salvos em: {arquivo.name}"
    )


# ============================================================
# FUNCAO PRINCIPAL
# ============================================================

def main():

    validar_configuracao()

    # Inverte o sentido para BH -> Vitoria.
    cidades = [
        ESTACOES[i]
        for i in reversed(SELECIONADAS)
    ]

    mercados = list(
        itertools.combinations(cidades, 2)
    )

    tarifas = carregar_tarifas(
        PASTA / "tarifas_vale.csv",
        mercados
    )

    print(
        "Sentido: Belo Horizonte -> Pedro Nolasco"
    )
    print(
        "Estacoes:", " -> ".join(cidades)
    )
    print(
        f"Classe: {CLASSE} | "
        f"Capacidade: {CAPACIDADE} | "
        f"Cenario: {CENARIO}"
    )

    if CENARIO == 3:

        simular_cenario_3(
            cidades, mercados, tarifas
        )

    else:

        if CENARIO == 2:
            demanda = carregar_demanda_fixa(mercados)
        else:
            demanda = None

        resultado = resolver_modelo(
            cidades, mercados, tarifas, demanda
        )

        imprimir_resultado(
            cidades, mercados, tarifas,
            demanda, resultado
        )


if __name__ == "__main__":
    main()
