# =============================================================================
# DOMINÂNCIA FISCAL NO BRASIL: UMA ANÁLISE DO PERÍODO PÓS-PANDEMIA
# Código de replicação do TCC
#
# Autor: Mateus Hickmann Strauss
# Orientador: Prof. Dr. Sabino da Silva Porto Júnior
# UFRGS - Faculdade de Ciências Econômicas - 2026
#
# Lê a base mensal (base_dominancia_fiscal.xlsx, aba "Base") e refaz os testes,
# as tabelas e as figuras do trabalho. As tabelas aparecem na tela com a mesma
# numeração do texto e as figuras são salvas na pasta "figuras".
# =============================================================================


# =============================================================================
# 0. PACOTES E CONFIGURAÇÕES
# =============================================================================
# Instalar uma vez: pip install numpy pandas scipy statsmodels arch matplotlib openpyxl

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats, linalg
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.coint_tables import c_sjt, c_sja
import arch
from arch.unitroot import PhillipsPerron

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

ARQUIVO = "base_dominancia_fiscal.xlsx"   # a base fica na mesma pasta do código
PASTA_FIG = "figuras"
os.makedirs(PASTA_FIG, exist_ok=True)

ALFA = 0.05       # nível de significância das decisões
P_MAX = 6         # defasagem máxima testada nos VAR
HORIZONTE = 10    # meses das funções de resposta ao impulso (mês 0 = mês do choque)

print("Python", sys.version.split()[0], "| numpy", np.__version__, "| pandas", pd.__version__,
      "| statsmodels", statsmodels.__version__, "| arch", arch.__version__,
      "| matplotlib", matplotlib.__version__)

# estilo das figuras
AZUL, VERMELHO = "#1f3a93", "#c0392b"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.linewidth": 0.6})


def etapa(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def mostrar(nome, tabela):
    print(f"\n{nome}")
    print(tabela.to_string())


def estrelas(p):
    # significância pelo p-valor: *** 1%, ** 5%, * 10%
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def virgula(ax):
    # números do eixo vertical com vírgula decimal
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: ("%g" % v).replace(".", ",")))


def eixo_anos(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def salvar(fig, nome):
    fig.savefig(os.path.join(PASTA_FIG, nome), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigura salva em {PASTA_FIG}/{nome}")


# =============================================================================
# 1. IMPORTAÇÃO DA BASE
# =============================================================================
etapa("ETAPA 1: Importação da base")

base = pd.read_excel(ARQUIVO, sheet_name="Base")
base.index = pd.to_datetime(base["data"].astype(str).str[:7], format="%Y-%m")
dados = base[["IR", "CDS", "IE", "ER"]].astype(float).asfreq("MS")
# IR: Selic (% a.a.) | CDS: CDS de 5 anos (pontos-base)
# IE: expectativa de inflação para 12 meses (% a.a.) | ER: câmbio (R$/US$)

print(f"Período: {dados.index[0]:%m/%Y} a {dados.index[-1]:%m/%Y} ({len(dados)} observações mensais)")
print("Valores ausentes:", int(dados.isna().sum().sum()))
print(dados.head())


# =============================================================================
# 2. FIGURA 1: VARIÁVEIS EM NÍVEL
# =============================================================================
etapa("ETAPA 2: Variáveis em nível (Figura 1)")

TITULOS = {"IR": "IR: Selic (% a.a.)", "CDS": "CDS de 5 anos (pontos-base)",
           "IE": "IE: expectativa de inflação (% a.a.)", "ER": r"ER: câmbio (R\$/US\$)"}

fig, eixos = plt.subplots(2, 2, figsize=(6.3, 4.6))
for ax, v in zip(eixos.ravel(), ["IR", "CDS", "IE", "ER"]):
    ax.plot(dados.index, dados[v], color=AZUL, lw=1.2)
    ax.set_title(TITULOS[v])
    eixo_anos(ax)
    virgula(ax)
fig.tight_layout()
salvar(fig, "figura1_niveis.png")


# =============================================================================
# 3. TESTES DE RAIZ UNITÁRIA (ADF E PHILLIPS-PERRON)
# =============================================================================
# Testes com constante e sem tendência.
# ADF: defasagens pelo BIC, com máximo de 12*(T/100)^(1/4) (Schwert, 1989).
# PP: largura de banda de Newey-West, 4*(T/100)^(2/9) (Newey e West, 1994).
# Asteriscos pelos valores críticos de MacKinnon (2010); p-valores de MacKinnon (1994).
etapa("ETAPA 3: Testes de raiz unitária")

T = len(dados)
MAXLAG_ADF = int(12 * (T / 100) ** 0.25)   # = 10
LAGS_PP = int(4 * (T / 100) ** (2 / 9))    # = 3


def estrelas_vc(estat, vc):
    # rejeição da raiz unitária a 1%, 5% e 10% pelos valores críticos
    if estat < vc["1%"]:
        return "***"
    if estat < vc["5%"]:
        return "**"
    if estat < vc["10%"]:
        return "*"
    return ""


def raiz_unitaria(serie):
    x = serie.dropna().values
    adf = adfuller(x, maxlag=MAXLAG_ADF, regression="c", autolag="BIC")
    pp = PhillipsPerron(x, lags=LAGS_PP, trend="c", test_type="tau")
    return {"Defasagens": adf[2],
            "ADF": f"{adf[0]:.3f}{estrelas_vc(adf[0], adf[4])}", "p-valor ADF": f"{adf[1]:.3f}",
            "PP": f"{pp.stat:.3f}{estrelas_vc(pp.stat, pp.critical_values)}",
            "p-valor PP": f"{pp.pvalue:.3f}"}


tab1 = pd.DataFrame({v: raiz_unitaria(dados[v]) for v in ["IR", "CDS", "IE", "ER"]}).T
mostrar("Tabela 1 - Testes de raiz unitária para as variáveis em nível", tab1)

# Selic: ADF rejeita pelo p-valor, mas não pelo valor crítico de 5% (resultado limítrofe)
adf_ir = adfuller(dados["IR"].values, maxlag=MAXLAG_ADF, regression="c", autolag="BIC")
print(f"\nSelic, ADF: estatística {adf_ir[0]:.3f}; valor crítico de 5% {adf_ir[4]['5%']:.3f}; "
      f"p-valor {adf_ir[1]:.3f}")

# Selic em primeira diferença (citado na seção 4.1)
mostrar("Selic em primeira diferença", pd.DataFrame({"ΔIR": raiz_unitaria(dados["IR"].diff())}).T)

# A Selic é tratada como I(0) e fica em nível; CDS, IE e ER entram em primeira diferença
dif = dados[["CDS", "IE", "ER"]].diff().dropna()

# Figura 2: variáveis em primeira diferença
ROTULOS_DIF = {"CDS": "ΔCDS (pontos-base por mês)", "IE": "ΔIE (p.p. por mês)",
               "ER": r"ΔER (R\$/US\$ por mês)"}
fig, eixos = plt.subplots(3, 1, figsize=(6.3, 5.25), sharex=True)
for ax, v in zip(eixos, ["CDS", "IE", "ER"]):
    ax.plot(dif.index, dif[v], color=AZUL, lw=1.1)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(ROTULOS_DIF[v])
    eixo_anos(ax)
    virgula(ax)
fig.tight_layout()
salvar(fig, "figura2_diferencas.png")

tab2 = pd.DataFrame({"Δ" + v: raiz_unitaria(dif[v]) for v in ["CDS", "IE", "ER"]}).T
mostrar("Tabela 2 - Testes de raiz unitária para as variáveis em primeira diferença", tab2)


# =============================================================================
# 4. TESTE DE COINTEGRAÇÃO DE JOHANSEN (CDS, IE E ER EM NÍVEL)
# =============================================================================
etapa("ETAPA 4: Teste de cointegração de Johansen")

# 4.1 Deriva: a média da primeira diferença é zero? (erros-padrão de Newey-West)
linhas = []
for v in ["CDS", "IE", "ER"]:
    d = dados[v].diff().dropna().values
    m = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS_PP})
    linhas.append([f"{m.params[0]:.4f}", f"{m.bse[0]:.4f}", f"{m.tvalues[0]:.2f}", f"{m.pvalues[0]:.3f}"])
tabA1 = pd.DataFrame(linhas, index=["ΔCDS", "ΔIE", "ΔER"],
                     columns=["Média", "Erro-padrão", "Estatística t", "p-valor"])
mostrar("Tabela A1 - Teste de deriva das séries I(1) (H0: média da 1ª diferença igual a zero)", tabA1)


# 4.2 Ordem do VAR em níveis. Mesma regra do VAR do modelo: parte da ordem do BIC
# e aumenta até o teste LM-F não rejeitar a ausência de autocorrelação (h = 1 a 4)
def teste_lm(res, h, so_h=False):
    # LM de autocorrelação dos resíduos do VAR (Breusch-Godfrey multivariado).
    # so_h=False: defasagens 1 a h em conjunto; so_h=True: só a defasagem h.
    # Retorna a estatística LM, o p-valor qui-quadrado e o p-valor da versão F de Rao
    # (Edgerton e Shukur, 1999)
    U = np.asarray(res.resid)
    X = np.asarray(res.endog_lagged)          # constante e defasagens do VAR
    n_obs, K = U.shape
    defs = [h] if so_h else list(range(1, h + 1))
    blocos = []
    for j in defs:
        b = np.zeros((n_obs, K))              # resíduos defasados, com zeros no início
        b[j:] = U[:-j]
        blocos.append(b)
    E = np.column_stack([X] + blocos)
    Ue = U - E @ np.linalg.lstsq(E, U, rcond=None)[0]
    Se, Sr = Ue.T @ Ue / n_obs, U.T @ U / n_obs
    lm = n_obs * (K - np.trace(np.linalg.solve(Sr, Se)))
    m = K * len(defs)
    s = np.sqrt((K**2 * m**2 - 4) / (K**2 + m**2 - 5))
    N = n_obs - X.shape[1] - m - 0.5 * (K - m + 1)
    gl1, gl2 = K * m, N * s - (0.5 * K * m - 1)
    F = ((np.linalg.det(Se) / np.linalg.det(Sr)) ** (-1 / s) - 1) * gl2 / gl1
    return lm, stats.chi2.sf(lm, K * m), stats.f.sf(F, gl1, gl2)


def criterios(tabela):
    # AIC, BIC e HQIC de VAR(0) a VAR(P_MAX), com constante e na mesma amostra
    sel = VAR(tabela).select_order(maxlags=P_MAX, trend="c")
    return pd.DataFrame({"AIC": sel.ics["aic"], "BIC": sel.ics["bic"], "HQIC": sel.ics["hqic"]})


def lm_por_ordem(tabela):
    # p-valores do LM-F, h = 1 a 4, para VAR(1) a VAR(P_MAX)
    linhas = []
    for p in range(1, P_MAX + 1):
        res = VAR(tabela).fit(p, trend="c")
        linhas.append([res.nobs] + [teste_lm(res, h)[2] for h in range(1, 5)])
    return pd.DataFrame(linhas, index=range(1, P_MAX + 1), columns=["Obs.", "h=1", "h=2", "h=3", "h=4"])


def escolher_ordem(crit, lm):
    p_bic = max(int(crit["BIC"].idxmin()), 1)
    for p in range(p_bic, P_MAX + 1):
        if lm.loc[p, ["h=1", "h=2", "h=3", "h=4"]].min() >= ALFA:
            return p
    return P_MAX


def marcar_minimo(crit):
    # critérios com 4 casas e * no menor valor de cada coluna
    tab = crit.apply(lambda col: col.map(lambda x: f"{x:.4f}"))
    for c in crit.columns:
        tab.loc[crit[c].idxmin(), c] += "*"
    return tab


niveis = dados[["CDS", "IE", "ER"]]
crit_c = criterios(niveis)
lm_c = lm_por_ordem(niveis)
P_COINT = escolher_ordem(crit_c, lm_c)

tabA2 = marcar_minimo(crit_c)
for h in ["h=1", "h=2", "h=3", "h=4"]:
    tabA2[h] = ["-"] + [f"{p:.3f}{estrelas(p)}" for p in lm_c[h]]
tabA2.index.name = "Defasagens"
mostrar("Tabela A2 - Seleção de defasagens do VAR em níveis de CDS, IE e ER "
        "(critérios e p-valores do LM-F)", tabA2)
print(f"Ordem escolhida: VAR({P_COINT}) em níveis")


# 4.3 Teste de Johansen
def johansen(Y, k, caso):
    # Regressão de posto reduzido (Johansen, 1988). k = defasagens das diferenças (p - 1).
    # caso 2: constante restrita ao vetor de cointegração; caso 3: constante irrestrita
    Y = np.asarray(Y, dtype=float)
    dY = np.diff(Y, axis=0)
    Z0, Z1 = dY[k:], Y[k:-1]                  # ΔY(t) e Y(t-1)
    n = len(Z0)
    Z2 = np.column_stack([dY[k - i:-i] for i in range(1, k + 1)]) if k > 0 else np.empty((n, 0))
    if caso == 2:
        Z1 = np.column_stack([Z1, np.ones(n)])
    else:
        Z2 = np.column_stack([Z2, np.ones(n)])

    def residuo(A, X):
        return A if X.shape[1] == 0 else A - X @ np.linalg.lstsq(X, A, rcond=None)[0]

    R0, R1 = residuo(Z0, Z2), residuo(Z1, Z2)
    S00, S01, S11 = R0.T @ R0 / n, R0.T @ R1 / n, R1.T @ R1 / n
    lam = linalg.eigh(S01.T @ np.linalg.solve(S00, S01), S11, eigvals_only=True)
    lam = np.clip(np.sort(lam)[::-1][:Y.shape[1]], 0, 1 - 1e-12)
    traco = np.array([-n * np.sum(np.log(1 - lam[r:])) for r in range(len(lam))])
    maximo = -n * np.log(1 - lam)
    return lam, traco, maximo, n


# valores críticos de 5% por m = número de tendências comuns (3 - r)
# caso 2: Osterwald-Lenum (1992); caso 3: MacKinnon, Haug e Michelis (1999), tabelados no statsmodels
VC_OL = {"traco": {1: 9.24, 2: 19.96, 3: 34.91}, "max": {1: 9.24, 2: 15.67, 3: 22.00}}


def vc5(caso, tipo, m):
    if caso == 2:
        return VC_OL[tipo][m]
    return (c_sjt if tipo == "traco" else c_sja)(m, 0)[1]


def pvalor_doornik(estat, m, tipo):
    # p-valor assintótico do caso 2 pela aproximação Gama de Doornik (1998)
    d1, d2 = float(m == 1), float(m == 2)
    if tipo == "traco":
        media = np.dot([2, 2.01, 0, 0.06, 0.05], [m * m, m, 1, d1, d2])
        var_ = np.dot([3, 3.6, 0.75, -0.4, -0.3], [m * m, m, 1, d1, d2])
    else:
        x = [m, 1, d1, d2, np.sqrt(m)]
        media = np.dot([5.9498, 0.43402, 0.04836, 0.018198, -2.3669], x)
        var_ = np.dot([2.2231, -7.9064, 0.58592, -0.034324, 12.058], x)
    return stats.gamma.sf(estat, a=media**2 / var_, scale=var_ / media)


def posto(estats, caso, tipo):
    # número de relações: primeira hipótese não rejeitada a 5%, a partir de r = 0
    for r, e in enumerate(estats):
        if e <= vc5(caso, tipo, len(estats) - r):
            return r
    return len(estats)


lam, traco, maximo, n_coint = johansen(niveis, P_COINT - 1, caso=2)
H0 = ["r = 0", "r <= 1", "r <= 2"]

tab3 = pd.DataFrame({"Traço": [f"{x:.2f}" for x in traco],
                     "VC 5% traço": [f"{vc5(2, 'traco', 3 - r):.2f}" for r in range(3)],
                     "Máx. autovalor": [f"{x:.2f}" for x in maximo],
                     "VC 5% máx.": [f"{vc5(2, 'max', 3 - r):.2f}" for r in range(3)]}, index=H0)
mostrar(f"Tabela 3 - Teste de cointegração de Johansen para CDS, IE e ER "
        f"(constante restrita, VAR({P_COINT}) em níveis)", tab3)

tabA3 = pd.DataFrame({"Autovalor": [f"{x:.4f}" for x in lam],
                      "Traço": [f"{x:.2f}" for x in traco],
                      "VC 5% traço": [f"{vc5(2, 'traco', 3 - r):.2f}" for r in range(3)],
                      "p-valor traço": [f"{pvalor_doornik(traco[r], 3 - r, 'traco'):.3f}" for r in range(3)],
                      "Máx. autovalor": [f"{x:.2f}" for x in maximo],
                      "VC 5% máx.": [f"{vc5(2, 'max', 3 - r):.2f}" for r in range(3)],
                      "p-valor máx.": [f"{pvalor_doornik(maximo[r], 3 - r, 'max'):.3f}" for r in range(3)]},
                     index=H0)
mostrar(f"Tabela A3 - Teste de Johansen: resultados completos ({n_coint} observações)", tabA3)

# 4.4 Especificações alternativas: caso 3 e ordem indicada pelo BIC
p_bic_c = max(int(crit_c["BIC"].idxmin()), 1)
especificacoes = [("Principal (caso 2)", P_COINT, 2), ("Caso 3", P_COINT, 3), ("Ordem do BIC", p_bic_c, 2)]
linhas = []
for nome, p, caso in especificacoes:
    _, tr, mx, _ = johansen(niveis, p - 1, caso)
    linhas.append([p, f"{tr[0]:.2f}", f"{vc5(caso, 'traco', 3):.2f}", posto(tr, caso, "traco"),
                   posto(mx, caso, "max")])
tabA4 = pd.DataFrame(linhas, index=[e[0] for e in especificacoes],
                     columns=["Defasagens", "Traço (r = 0)", "VC 5%", "Posto (traço)",
                              "Posto (máx. autovalor)"])
mostrar("Tabela A4 - Teste de Johansen em especificações alternativas", tabA4)


# =============================================================================
# 5. SELEÇÃO DE DEFASAGENS DO VAR
# =============================================================================
# Sistema do VAR: Selic em nível e CDS, IE e ER em primeira diferença,
# na ordem de Cholesky da teoria (IR, CDS, IE, ER)
etapa("ETAPA 5: Seleção de defasagens do VAR")

sistema = pd.concat([dados["IR"], dif], axis=1).dropna().asfreq("MS")
sistema.columns = ["IR", "ΔCDS", "ΔIE", "ΔER"]
COLS = list(sistema.columns)
K = len(COLS)
print(f"Sistema: {sistema.index[0]:%m/%Y} a {sistema.index[-1]:%m/%Y} ({len(sistema)} observações)")

crit_v = criterios(sistema)
lm_v = lm_por_ordem(sistema)
P_VAR = escolher_ordem(crit_v, lm_v)

tab4 = marcar_minimo(crit_v)
tab4.index.name = "Defasagens"
mostrar("Tabela 4 - Critérios de seleção de defasagens do VAR (* menor valor)", tab4)

tabB1 = pd.DataFrame({"Observações": lm_v["Obs."]})
for h in ["h=1", "h=2", "h=3", "h=4"]:
    tabB1[h] = [f"{p:.3f}{estrelas(p)}" for p in lm_v[h]]
tabB1.index.name = "Defasagens"
mostrar("Tabela B1 - Teste LM-F de autocorrelação dos resíduos por ordem do VAR (p-valores)", tabB1)
print(f"Ordem escolhida: VAR({P_VAR}), o primeiro sem autocorrelação a 5% em h = 1 a 4")


# =============================================================================
# 6. ESTIMAÇÃO DO VAR(4)
# =============================================================================
etapa(f"ETAPA 6: Estimação do VAR({P_VAR})")

var = VAR(sistema).fit(P_VAR, trend="c")
print(f"Amostra de estimação: {sistema.index[P_VAR]:%m/%Y} a {sistema.index[-1]:%m/%Y} "
      f"({var.nobs} observações, {K * P_VAR + 1} coeficientes por equação)")
print(f"Média da Selic no período: {dados['IR'].mean():.2f}% a.a.")   # citada na justificativa da constante


def nome_regressor(r):
    if r == "const":
        return "Constante"
    lag, v = r.split(".", 1)                  # ex.: "L1.ΔCDS" vira "ΔCDS(-1)"
    return f"{v}(-{lag[1:]})"


tabC1 = pd.DataFrame(index=[nome_regressor(r) for r in var.params.index])
for eq in COLS:
    tabC1[eq] = [f"{c:.4f}{estrelas(p)} ({e:.4f})"
                 for c, e, p in zip(var.params[eq], var.bse[eq], var.pvalues[eq])]
y_obs = var.fittedvalues + var.resid
r2 = 1 - (var.resid**2).sum() / ((y_obs - y_obs.mean())**2).sum()
tabC1.loc["R²"] = [f"{r2[eq]:.3f}" for eq in COLS]
tabC1.loc["Observações"] = [str(var.nobs)] * K
mostrar(f"Tabela C1 - Estimativas das equações do VAR({P_VAR}) (erros-padrão entre parênteses)", tabC1)

# Teste de Wald conjunto: as quatro constantes iguais a zero
c = var.params.loc["const"].values
Z = np.asarray(var.endog_lagged)
V_c = np.linalg.inv(Z.T @ Z)[0, 0] * np.asarray(var.sigma_u)
wald_c = c @ np.linalg.solve(V_c, c)
print(f"\nTeste de Wald das constantes: qui-quadrado = {wald_c:.2f}, gl = {K}, "
      f"p-valor = {stats.chi2.sf(wald_c, K):.3f}")


# =============================================================================
# 7. DIAGNÓSTICO DO VAR(4)
# =============================================================================
etapa(f"ETAPA 7: Diagnóstico do VAR({P_VAR})")

# 7.1 Autocorrelação: LM na defasagem h (1 a 12) e LM-F conjunto até h (1 a 6)
linhas = []
for h in range(1, 13):
    lm_h, p_qui, p_f = teste_lm(var, h, so_h=True)
    p_conj = f"{teste_lm(var, h)[2]:.3f}" if h <= 6 else "-"
    linhas.append([f"{lm_h:.2f}", f"{p_qui:.3f}{estrelas(p_qui)}", f"{p_f:.3f}{estrelas(p_f)}", p_conj])
tabD1 = pd.DataFrame(linhas, index=pd.Index(range(1, 13), name="h"),
                     columns=["LM", "p-valor (qui-quadrado)", "p-valor (F)", "p-valor conjunto 1 a h (F)"])
mostrar(f"Tabela D1 - Teste LM de autocorrelação dos resíduos do VAR({P_VAR})", tabD1)

# 7.2 Normalidade: teste de Lütkepohl, com os resíduos ortogonalizados por Cholesky
U = var.resid.values - var.resid.values.mean(axis=0)
n_res = len(U)
W = U @ np.linalg.inv(np.linalg.cholesky(U.T @ U / n_res)).T
chi_a = n_res * (W**3).mean(axis=0) ** 2 / 6               # assimetria
chi_c = n_res * ((W**4).mean(axis=0) - 3) ** 2 / 24        # curtose
linhas = [[chi_a[i], stats.chi2.sf(chi_a[i], 1), chi_c[i], stats.chi2.sf(chi_c[i], 1),
           chi_a[i] + chi_c[i], stats.chi2.sf(chi_a[i] + chi_c[i], 2)] for i in range(K)]
linhas.append([chi_a.sum(), stats.chi2.sf(chi_a.sum(), K), chi_c.sum(), stats.chi2.sf(chi_c.sum(), K),
               chi_a.sum() + chi_c.sum(), stats.chi2.sf(chi_a.sum() + chi_c.sum(), 2 * K)])
tabE1 = pd.DataFrame([[f"{l[0]:.3f}", f"{l[1]:.3f}{estrelas(l[1])}", f"{l[2]:.3f}",
                       f"{l[3]:.3f}{estrelas(l[3])}", f"{l[4]:.3f}", f"{l[5]:.3f}{estrelas(l[5])}"]
                      for l in linhas],
                     index=COLS + ["Conjunto"],
                     columns=["Assimetria", "p-valor", "Curtose", "p-valor ", "Jarque-Bera", "p-valor  "])
mostrar(f"Tabela E1 - Teste de normalidade dos resíduos do VAR({P_VAR}) (H0: normalidade)", tabE1)

# 7.3 Estabilidade: raízes inversas (autovalores da matriz companheira)
comp = np.zeros((K * P_VAR, K * P_VAR))
comp[:K, :] = np.hstack(var.coefs)
comp[K:, :-K] = np.eye(K * (P_VAR - 1))
raizes = np.linalg.eigvals(comp)
raizes = raizes[np.argsort(-np.abs(raizes))]
tabF1 = pd.DataFrame({"Parte real": [f"{x:.4f}" for x in raizes.real],
                      "Parte imaginária": [f"{x:.4f}" for x in raizes.imag],
                      "Módulo": [f"{x:.4f}" for x in np.abs(raizes)]},
                     index=pd.Index(range(1, len(raizes) + 1), name="Raiz"))
mostrar(f"Tabela F1 - Raízes inversas do polinômio característico do VAR({P_VAR})", tabF1)
print("Maior módulo:", f"{np.abs(raizes).max():.4f}", "| VAR estável:", var.is_stable())


# =============================================================================
# 8. CAUSALIDADE DE GRANGER
# =============================================================================
etapa("ETAPA 8: Causalidade de Granger")


# 8.1 Bivariada: teste F com 1 a 8 defasagens (regressão de y em constante,
# defasagens de y e defasagens de x)
def granger_bivariado(y, x, p):
    d = pd.concat([y, x], axis=1).dropna().values
    n = len(d)
    alvo = d[p:, 0]
    def_y = np.column_stack([d[p - j:n - j, 0] for j in range(1, p + 1)])
    def_x = np.column_stack([d[p - j:n - j, 1] for j in range(1, p + 1)])
    um = np.ones((n - p, 1))
    X_r, X_u = np.column_stack([um, def_y]), np.column_stack([um, def_y, def_x])
    ssr_r = np.sum((alvo - X_r @ np.linalg.lstsq(X_r, alvo, rcond=None)[0]) ** 2)
    ssr_u = np.sum((alvo - X_u @ np.linalg.lstsq(X_u, alvo, rcond=None)[0]) ** 2)
    gl = (n - p) - X_u.shape[1]
    F = ((ssr_r - ssr_u) / p) / (ssr_u / gl)
    return stats.f.sf(F, p, gl)


p_biv = {}
for y in COLS:
    for x in COLS:
        if x != y:
            p_biv[(x, y)] = [granger_bivariado(sistema[y], sistema[x], p) for p in range(1, 9)]
tabG1 = pd.DataFrame([[f"{p:.3f}{estrelas(p)}" for p in p_biv[par]] for par in p_biv],
                     index=[f"{x} não Granger-causa {y}" for x, y in p_biv], columns=range(1, 9))
mostrar("Tabela G1 - Causalidade de Granger bivariada (p-valores do teste F, 1 a 8 defasagens)", tabG1)

# 8.2 Multivariada: teste de Wald no VAR(4)
p_multi, linhas, nomes = {}, [], []
for y in COLS:
    outras = [x for x in COLS if x != y]
    for x in outras + ["Todas"]:
        t = var.test_causality(y, outras if x == "Todas" else [x], kind="wald")
        p_multi[(x, y)] = t.pvalue
        nomes.append((y, x))
        linhas.append([f"{t.test_statistic:.2f}", int(np.atleast_1d(t.df)[0]),
                       f"{t.pvalue:.3f}{estrelas(t.pvalue)}"])
tabH1 = pd.DataFrame(linhas, index=pd.MultiIndex.from_tuples(nomes, names=["Dependente", "Excluída"]),
                     columns=["Qui-quadrado", "gl", "p-valor"])
mostrar(f"Tabela H1 - Causalidade de Granger multivariada no VAR({P_VAR})", tabH1)


# 8.3 Resumo (Tabela 5). Bivariado: há causalidade quando H0 é rejeitada a 5% em
# 5 ou mais das 8 defasagens; não há com até 2 rejeições; inconclusivo com 3 ou 4
def resumo_bivariado(n_rej):
    return "rejeita" if n_rej >= 5 else "não rejeita" if n_rej <= 2 else "inconclusivo"


linhas, nomes = [], []
for y in COLS:
    for x in [x for x in COLS if x != y] + ["Todas"]:
        if x == "Todas":
            nomes.append(f"Todas não Granger-causam {y}")
            biv = "-"
        else:
            nomes.append(f"{x} não Granger-causa {y}")
            n_rej = sum(p < ALFA for p in p_biv[(x, y)])
            biv = f"{n_rej} de 8 ({resumo_bivariado(n_rej)})"
        linhas.append([biv] + tabH1.loc[(y, x), ["Qui-quadrado", "p-valor"]].tolist())
tab5 = pd.DataFrame(linhas, index=nomes, columns=["Bivariado: rejeições a 5%", "Qui-quadrado", "p-valor"])
mostrar("Tabela 5 - Testes de causalidade de Granger", tab5)

# 8.4 Ordem de Granger: da variável menos Granger-causada para a mais causada (multivariado, 5%);
# empate decidido pelo p-valor do teste conjunto ("Todas"), do maior para o menor
contagem = {y: sum(p_multi[(x, y)] < ALFA for x in COLS if x != y) for y in COLS}
ORDEM_GRANGER = sorted(COLS, key=lambda y: (contagem[y], -p_multi[("Todas", y)]))
print("\nOrdem da teoria: ", ", ".join(COLS))
print("Ordem de Granger:", ", ".join(ORDEM_GRANGER))


# =============================================================================
# 9. CORRELAÇÃO CONTEMPORÂNEA ENTRE OS RESÍDUOS
# =============================================================================
etapa("ETAPA 9: Correlação contemporânea entre os resíduos")

sigma = np.asarray(var.sigma_u)
dp = np.sqrt(np.diag(sigma))
tabI1 = pd.DataFrame(sigma / np.outer(dp, dp), index=COLS, columns=COLS)
tabI1 = tabI1.apply(lambda col: col.map(lambda x: f"{x:.3f}"))
mostrar(f"Tabela I1 - Correlação contemporânea entre os resíduos do VAR({P_VAR})", tabI1)


# =============================================================================
# 10. FUNÇÕES DE RESPOSTA AO IMPULSO
# =============================================================================
# Choque de um desvio-padrão, identificado por Cholesky, com bandas de ±2 erros-padrão
# (assintóticos). Para CDS, IE e ER as respostas são acumuladas (efeito sobre o nível).
# A resposta é significativa no mês em que as duas bandas ficam do mesmo lado do zero.
etapa("ETAPA 10: Funções de resposta ao impulso")


def fir(ordem):
    res = VAR(sistema[ordem]).fit(P_VAR, trend="c")   # mesma forma reduzida, outra ordem de Cholesky
    irf = res.irf(HORIZONTE)
    resp, ep = irf.orth_irfs.copy(), irf.stderr(orth=True).copy()
    em_dif = [i for i, v in enumerate(ordem) if v.startswith("Δ")]
    resp[:, em_dif, :] = irf.orth_cum_effects[:, em_dif, :]
    ep[:, em_dif, :] = irf.cum_effect_stderr(orth=True)[:, em_dif, :]
    volta = [ordem.index(v) for v in COLS]            # volta para a ordem IR, ΔCDS, ΔIE, ΔER
    return resp[:, volta][:, :, volta], ep[:, volta][:, :, volta]   # [mês, resposta, choque]


resp_t, ep_t = fir(COLS)              # ordem da teoria: IR, CDS, IE, ER
resp_g, ep_g = fir(ORDEM_GRANGER)     # ordem de Granger


# 10.1 Quadro 1: meses com resposta significativa nos pares relevantes para a hipótese
def meses_texto(m):
    if len(m) > 2 and m[-1] - m[0] == len(m) - 1:
        return f"meses {m[0]} a {m[-1]}"
    return ("mês " if len(m) == 1 else "meses ") + ", ".join(str(h) for h in m)


def avaliar(resp, ep, choque, resposta):
    i, j = COLS.index(resposta), COLS.index(choque)
    m = [h for h in range(HORIZONTE + 1) if abs(resp[h, i, j]) > 2 * ep[h, i, j]]
    if not m:
        return "não significativa"
    sinais = {np.sign(resp[h, i, j]) for h in m}
    if len(sinais) > 1:
        return f"sinais opostos ({meses_texto(m)})"
    if resposta == "ΔER":                             # câmbio: positivo = depreciação
        sentido = "depreciação" if sinais == {1.0} else "apreciação"
    else:
        sentido = "aumento" if sinais == {1.0} else "queda"
    return f"{sentido} ({meses_texto(m)})"


# (choque, resposta)
PARES = [("IR", "ΔCDS"), ("IR", "ΔIE"), ("IR", "ΔER"), ("ΔCDS", "ΔER"), ("ΔIE", "ΔER")]
quadro1 = pd.DataFrame([[avaliar(resp_t, ep_t, c, r), avaliar(resp_g, ep_g, c, r)] for c, r in PARES],
                       index=[f"{c.replace('Δ', '')} -> {r.replace('Δ', '')}" for c, r in PARES],
                       columns=["Ordem da teoria", "Ordem de Granger"])
mostrar("Quadro 1 - Síntese das funções de resposta ao impulso relevantes para a hipótese", quadro1)

# 10.2 Figuras 3 e 4
UNID = {"IR": "p.p.", "ΔCDS": "pontos-base", "ΔIE": "p.p.", "ΔER": r"R\$/US\$"}
# Figura 4: na ordem de Granger, só as respostas significativas na ordem da teoria (Quadro 1)
FIG4 = [("IR", "ΔER"), ("IR", "ΔIE"), ("ΔCDS", "ΔER"), ("ΔIE", "ΔER")]   # (choque, resposta)

# mesma escala vertical em cada linha das duas figuras, para comparar as ordens
ylim = {}
for i, r in enumerate(COLS):
    inf = [resp_t[:, i, :] - 2 * ep_t[:, i, :]] + [resp_g[:, i, COLS.index(c)] - 2 * ep_g[:, i, COLS.index(c)]
                                                   for c, rr in FIG4 if rr == r]
    sup = [resp_t[:, i, :] + 2 * ep_t[:, i, :]] + [resp_g[:, i, COLS.index(c)] + 2 * ep_g[:, i, COLS.index(c)]
                                                   for c, rr in FIG4 if rr == r]
    lo = min(0, min(a.min() for a in inf))
    hi = max(0, max(a.max() for a in sup))
    ylim[r] = (lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))


def painel(ax, resp, ep, i, j, titulo, limites):
    h = np.arange(HORIZONTE + 1)
    ax.plot(h, resp[:, i, j], color=AZUL, lw=1.4)
    ax.plot(h, resp[:, i, j] + 2 * ep[:, i, j], color=VERMELHO, ls="--", lw=0.9)
    ax.plot(h, resp[:, i, j] - 2 * ep[:, i, j], color=VERMELHO, ls="--", lw=0.9)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(titulo, fontsize=7.5)
    ax.set_xticks(h[::2])
    ax.set_xlim(0, HORIZONTE)
    ax.set_ylim(*limites)
    virgula(ax)


def legenda(fig):
    fig.legend([Line2D([], [], color=AZUL, lw=1.4), Line2D([], [], color=VERMELHO, ls="--", lw=0.9)],
               ["resposta", "±2 erros-padrão"], loc="lower center", ncol=2, frameon=False, fontsize=7.5)


# Figura 3: ordem da teoria (IR, CDS, IE, ER); colunas = choque, linhas = resposta
fig, eixos = plt.subplots(K, K, figsize=(6.5, 6.6), sharex=True)
for i, r in enumerate(COLS):
    for j, s in enumerate(COLS):
        painel(eixos[i, j], resp_t, ep_t, i, j,
               f"Resposta de {r.replace('Δ', '')} a {s.replace('Δ', '')}", ylim[r])
        if j == 0:
            eixos[i, j].set_ylabel(UNID[r], fontsize=7.5)
        if i == K - 1:
            eixos[i, j].set_xlabel("meses", fontsize=7.5)
legenda(fig)
fig.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "figura3_fir_ordem_teoria.png")

# Figura 4: ordem de Granger (ER, IE, IR, CDS)
fig, eixos = plt.subplots(2, 2, figsize=(6.3, 5.0))
for ax, (c, r) in zip(eixos.ravel(), FIG4):
    painel(ax, resp_g, ep_g, COLS.index(r), COLS.index(c),
           f"Resposta de {r.replace('Δ', '')} a {c.replace('Δ', '')}", ylim[r])
    ax.set_ylabel(UNID[r], fontsize=7.5)
    ax.set_xlabel("meses", fontsize=7.5)
legenda(fig)
fig.tight_layout(rect=(0, 0.05, 1, 1))
salvar(fig, "figura4_fir_ordem_granger.png")

print("\nFim da replicação.")
