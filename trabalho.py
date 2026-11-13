"""Alocacao de assentos EFVM: BH -> Pedro Nolasco (Vitoria).

Requisitos: pip install pulp numpy
Coloque tarifas_vale.csv na mesma pasta deste arquivo.
Tarifas de 10/12/2025; hipotese de simetria entre sentidos.
Demandas e capacidade sao hipoteses de estudo, nao observacoes operacionais.
"""
import csv
import itertools
import statistics
import time
from pathlib import Path

import numpy as np
import pulp

PASTA = Path(__file__).resolve().parent
CAPACIDADE = 2250  # referencia historica fornecida; classe economica
CENARIO = 1         # 1: ilimitada; 2: fixa; 3: simulacoes
CLASSE = 'economica'
REPETICOES = 100
SEMENTE = 42

# Indices seguem a ordem ORIGINAL do CSV: ES -> MG.
# A simulacao reverte essa ordem automaticamente para BH -> ES.
ESTACOES = [
    'Pedro Nolasco', 'Flexal', 'Fundão', 'Aricanga', 'Piraqueaçu',
    'Colatina', 'Itapina', 'Mascarenhas', 'Baixo Guandu', 'Aimorés',
    'Resplendor', 'Crenaque', 'Conselheiro Pena', 'Barra do Cuieté',
    'São Tomé do Rio Doce', 'Tumiritinga', 'Governador Valadares',
    'Pedra Corrida', 'Periquito', 'Frederico Sellow', 'Ipaba',
    'Intendente Câmara', 'Mário Carvalho', 'Antônio Dias',
    'Desembargador Drumond', 'João Monlevade', 'Rio Piracicaba',
    'Dois Irmãos', 'Belo Horizonte'
]
SELECIONADAS = [0, 5, 16, 21, 28]  # BH, Intendente, GV, Colatina, Pedro Nolasco

# Valores hipoteticos para as cinco estacoes de exemplo, sentido BH -> ES.
DEMANDA_EXEMPLO = {
    ('Belo Horizonte', 'Intendente Câmara'): 800,
    ('Belo Horizonte', 'Governador Valadares'): 400,
    ('Belo Horizonte', 'Colatina'): 380,
    ('Belo Horizonte', 'Pedro Nolasco'): 600,
    ('Intendente Câmara', 'Governador Valadares'): 650,
    ('Intendente Câmara', 'Colatina'): 250,
    ('Intendente Câmara', 'Pedro Nolasco'): 240,
    ('Governador Valadares', 'Colatina'): 500,
    ('Governador Valadares', 'Pedro Nolasco'): 320,
    ('Colatina', 'Pedro Nolasco'): 450,
}

# Simulacoes: valores uniformes (pessoas por mercado O-D).
# A quarta familia concentra maior demanda nos mercados com polos regionais.
PERFIS = {
    'baixa': (50, 250),
    'moderada': (200, 600),
    'alta': (600, 1400),
    'polos_regionais': (100, 450),
}
POLOS = {'Belo Horizonte', 'Governador Valadares', 'Intendente Câmara', 'Pedro Nolasco'}
FATOR_POLOS = 2.0


def validar_configuracao():
    if CENARIO not in (1, 2, 3):
        raise ValueError('CENARIO deve ser 1, 2 ou 3')
    if CLASSE not in ('economica', 'executiva'):
        raise ValueError('CLASSE invalida')
    if not isinstance(CAPACIDADE, int) or CAPACIDADE <= 0:
        raise ValueError('CAPACIDADE deve ser inteiro positivo')
    if len(SELECIONADAS) < 2 or SELECIONADAS != sorted(set(SELECIONADAS)):
        raise ValueError('SELECIONADAS precisa ter >= 2 indices distintos e crescentes')
    if not all(0 <= i < len(ESTACOES) for i in SELECIONADAS):
        raise ValueError('Indice de estacao invalido')
    if REPETICOES < 1:
        raise ValueError('REPETICOES deve ser positiva')


def carregar_tarifas(arquivo, mercados):
    """O CSV esta em ES->MG. Assumimos simetria tarifaria no sentido inverso."""
    tabela = {}
    with arquivo.open(encoding='utf-8-sig', newline='') as f:
        for linha in csv.DictReader(f):
            chave = (linha['origem'].strip(), linha['destino'].strip())
            if chave in tabela:
                raise ValueError(f'Tarifa duplicada: {chave}')
            tabela[chave] = float(linha[CLASSE].replace(',', '.'))
    tarifas = {}
    for a, b in mercados:
        if (a, b) in tabela:
            tarifas[a, b] = tabela[a, b]
        elif (b, a) in tabela:
            tarifas[a, b] = tabela[b, a]
        else:
            raise ValueError(f'Tarifa nao encontrada para {a} -> {b}')
    return tarifas


def carregar_demanda_fixa(mercados):
    arquivo = PASTA / 'demandas_cenario2_BH_Vitoria.csv'
    if not arquivo.exists():
        with arquivo.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['origem', 'destino', 'demanda'])
            for a, b in mercados:
                writer.writerow([a, b, DEMANDA_EXEMPLO.get((a, b), '')])
        print(f'Criado {arquivo.name}. Preencha demandas vazias antes de executar.')
    dados = {}
    with arquivo.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            a, b = row['origem'].strip(), row['destino'].strip()
            valor = row['demanda'].strip()
            if not valor:
                continue
            try:
                numero = int(valor)
            except ValueError as exc:
                raise ValueError(f'Demanda deve ser inteiro: {a} -> {b}: {valor}') from exc
            if numero < 0:
                raise ValueError(f'Demanda negativa: {a} -> {b}')
            if (a, b) in dados:
                raise ValueError(f'Demanda duplicada: {a} -> {b}')
            dados[a, b] = numero
    faltantes = [od for od in mercados if od not in dados]
    if faltantes:
        raise ValueError(f'Preencha {arquivo.name} para os mercados: {faltantes}')
    return {od: dados[od] for od in mercados}


def gerar_demanda(mercados, perfil, rng):
    minimo, maximo = PERFIS[perfil]
    dados = {}
    for a, b in mercados:
        valor = int(rng.integers(minimo, maximo + 1))
        if perfil == 'polos_regionais' and (a in POLOS or b in POLOS):
            valor = round(valor * FATOR_POLOS)
        dados[a, b] = valor
    return dados


def resolver_modelo(cidades, mercados, tarifas, demanda=None):
    """Cria um novo modelo por execucao, sem reutilizar restricoes antigas."""
    modelo = pulp.LpProblem('Alocacao_EFVM', pulp.LpMaximize)
    x = {od: pulp.LpVariable(f'x_{i}_{j}', lowBound=0, cat='Integer')
         for i, a in enumerate(cidades) for j, b in enumerate(cidades) if i < j
         for od in [(a, b)]}
    pos = {nome: i for i, nome in enumerate(cidades)}
    modelo += pulp.lpSum(tarifas[od] * x[od] for od in mercados)
    for s in range(len(cidades) - 1):
        modelo += (pulp.lpSum(x[a, b] for a, b in mercados
                             if pos[a] <= s < pos[b]) <= CAPACIDADE,
                   f'capacidade_{s}')
    if demanda is not None:
        for a, b in mercados:
            modelo += x[a, b] <= demanda[a, b], f'demanda_{pos[a]}_{pos[b]}'
    inicio = time.perf_counter()
    status = modelo.solve(pulp.PULP_CBC_CMD(msg=False))
    tempo = time.perf_counter() - inicio
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f'Solver retornou: {pulp.LpStatus[status]}')
    vendas = {od: int(round(pulp.value(x[od]))) for od in mercados}
    ocupacoes = [sum(vendas[a, b] for a, b in mercados if pos[a] <= s < pos[b])
                 for s in range(len(cidades) - 1)]
    receita = sum(tarifas[od] * vendas[od] for od in mercados)
    return {
        'receita': receita,
        'passagens': sum(vendas.values()),
        'ocupacoes': ocupacoes,
        'ocupacao_media_pct': 100 * statistics.mean(ocupacoes) / CAPACIDADE,
        'atendimento_pct': (100 * sum(vendas.values()) / sum(demanda.values())
                            if demanda is not None and sum(demanda.values()) > 0 else None),
        'tempo': tempo,
        'vendas': vendas,
    }


def imprimir_resultado(cidades, mercados, tarifas, demanda, resultado):
    print(f"Receita maxima: R$ {resultado['receita']:,.2f}")
    print(f"Tempo de resolucao: {resultado['tempo']:.6f} s")
    print(f"Passagens vendidas: {resultado['passagens']}")
    print('\nOrigem -> Destino | Vendidas | Demanda | Tarifa')
    for a, b in mercados:
        q = resultado['vendas'][a, b]
        #if q:
        d = demanda[a, b] if demanda is not None else 'ilimitada'
        print(f'{a} -> {b} | {q} | {d} | R$ {tarifas[a, b]:.2f}')
    print('\nOcupacao por trecho:')
    for s, q in enumerate(resultado['ocupacoes']):
        print(f'{cidades[s]} -> {cidades[s+1]}: '
              f'{q}/{CAPACIDADE} ({100*q/CAPACIDADE:.1f}%)')
    if demanda is not None:
        print(f"Atendimento da demanda: {resultado['passagens']}/{sum(demanda.values())} "
              f"({resultado['atendimento_pct']:.1f}%)" if resultado['atendimento_pct'] is not None
              else 'Atendimento da demanda: demanda total zero')


def simular_cenario_3(cidades, mercados, tarifas):
    # Uma semente por perfil permite reproduzir integralmente o experimento.
    sementes = np.random.SeedSequence(SEMENTE).spawn(len(PERFIS))
    registros = []
    print(f'\nCenario 3 | {len(cidades)} estacoes | {len(mercados)} variaveis | '
          f'{REPETICOES} repeticoes por perfil')
    for (perfil, _), seed in zip(PERFIS.items(), sementes):
        rng = np.random.default_rng(seed)
        resultados = []
        for repeticao in range(1, REPETICOES + 1):
            demanda = gerar_demanda(mercados, perfil, rng)
            r = resolver_modelo(cidades, mercados, tarifas, demanda)
            registro = {
                'perfil': perfil, 'repeticao': repeticao,
                'receita': r['receita'], 'passagens': r['passagens'],
                'demanda_total': sum(demanda.values()),
                'ocupacao_media_pct': r['ocupacao_media_pct'],
                'atendimento_pct': r['atendimento_pct'],
                'tempo_s': r['tempo'],
            }
            registros.append(registro)
            resultados.append(registro)
        def media(campo):
            return statistics.mean(r[campo] for r in resultados)
        print(f'\nPerfil: {perfil}')
        print(f"Receita media: R$ {media('receita'):,.2f}")
        print(f"Desvio padrao da receita: R$ {statistics.stdev(r['receita'] for r in resultados):,.2f}"
              if REPETICOES > 1 else 'Desvio padrao: N/A (uma repeticao)')
        print(f"Passagens medias: {media('passagens'):.1f}")
        print(f"Ocupacao media: {media('ocupacao_media_pct'):.1f}%")
        print(f"Atendimento medio: {media('atendimento_pct'):.1f}%")
        print(f"Tempo mediano: {statistics.median(r['tempo_s'] for r in resultados):.4f} s")

    arquivo = PASTA / 'resultados_cenario3.csv'
    with arquivo.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(registros[0]))
        writer.writeheader()
        writer.writerows(registros)
    print(f'\nResultados das simulacoes salvos em: {arquivo.name}')


def main():
    validar_configuracao()
    cidades = [ESTACOES[i] for i in reversed(SELECIONADAS)]
    mercados = list(itertools.combinations(cidades, 2))
    tarifas = carregar_tarifas(PASTA / 'tarifas_vale.csv', mercados)
    print('Sentido: Belo Horizonte -> Pedro Nolasco (Grande Vitoria)')
    print('Estacoes:', ' -> '.join(cidades))
    print(f'Classe: {CLASSE} | Capacidade: {CAPACIDADE} | Cenario: {CENARIO}')
    if CENARIO == 3:
        simular_cenario_3(cidades, mercados, tarifas)
    else:
        demanda = carregar_demanda_fixa(mercados) if CENARIO == 2 else None
        resultado = resolver_modelo(cidades, mercados, tarifas, demanda)
        imprimir_resultado(cidades, mercados, tarifas, demanda, resultado)


if __name__ == '__main__':
    main()
