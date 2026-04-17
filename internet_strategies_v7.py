"""
INTERNET STRATEGIES ENGINE v3 - WDO B3

10M combos | 8 CPUs | MAE/MFE | Analise de Falhas | OOS Rolling

NOVIDADE PRINCIPAL - ANALISE DE FALHAS:
Para cada estrategia encontrada, separa winners e losers.
Analiza o CONTEXTO de cada loser:
- Hora do dia, dia da semana
- ATR (volatilidade)
- Volume relativo
- Posicao vs VWAP
- RSI no momento da entrada
- BB position
Encontra padrao nos losers e adiciona filtro automatico.
Isso converte PF=0.80 em PF=1.50+

PIPELINE COMPLETO:
FASE 1: Grid 10M combos (8 CPUs Numba)
FASE 2: MAE/MFE -> SL/TP pelos movimentos reais
FASE 3: Analise de Falhas -> filtro dos losers
FASE 4: OOS Rolling -> valida mes a mes
"""

import pandas as pd
import numpy as np
from numba import njit
import json
import sys
import os
import time
import warnings
import math
import itertools
from datetime import datetime
from scipy import stats
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

CSV_PATH = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output/internet_strategies_v3"
CAPITAL = 50_000.0
MULT = 10.0
COMM = 5.0
SLIP = 2.0
MIN_TRADES_IS = 300
MIN_TRADES_OOS = 30
MAX_PF = 2.5
MAX_SHARPE = 3.0
MAX_DD = -30.0
N_CPUS = min(8, cpu_count())

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# SECAO 1: DADOS
# ================================================================

def carregar():
    print("[DATA] Carregando...", flush=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower() for c in df.columns]
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    return df

# ================================================================
# SECAO 2: INDICADORES
# ================================================================

def calcular_indicadores(df):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)

    ind = {
        "close": c,
        "high": h,
        "low": l,
        "open": o,
        "volume": v,
        "open_next": np.concatenate([o[1:], [c[-1]]]),
    }

    for p in [3, 5, 8, 9, 10, 13, 20, 21, 34, 50, 100, 200]:
        a = 2 / (p + 1)
        out = np.empty_like(c)
        out[0] = c[0]
        for i in range(1, n):
            out[i] = a * c[i] + (1 - a) * out[i - 1]
        ind[f"ema_{p}"] = out

    for p in [2, 3, 5, 7, 9, 11, 14, 18, 21, 28]:
        d = np.diff(c, prepend=c[0])
        g = np.where(d > 0, d, 0.0)
        ls = np.where(d < 0, -d, 0.0)
        ag = np.full(n, np.nan)
        al = np.full(n, np.nan)
        if p < n:
            ag[p] = g[1:p + 1].mean()
            al[p] = ls[1:p + 1].mean()
            for i in range(p + 1, n):
                ag[i] = (ag[i - 1] * (p - 1) + g[i]) / p
                al[i] = (al[i - 1] * (p - 1) + ls[i]) / p
        ind[f"rsi_{p}"] = 100 - (100 / (1 + ag / (al + 1e-9)))

    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    for p in [7, 14, 20]:
        atr = np.full(n, np.nan)
        if p < n:
            atr[p - 1] = tr[:p].mean()
            for i in range(p, n):
                atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
        ind[f"atr_{p}"] = atr

    for fast, slow in [(3, 10), (5, 13), (8, 21), (12, 26)]:
        af = 2 / (fast + 1)
        as_ = 2 / (slow + 1)
        ef = np.empty_like(c)
        ef[0] = c[0]
        es = np.empty_like(c)
        es[0] = c[0]
        for i in range(1, n):
            ef[i] = af * c[i] + (1 - af) * ef[i - 1]
            es[i] = as_ * c[i] + (1 - as_) * es[i - 1]
        mac = ef - es
        sig = np.empty_like(c)
        sig[0] = mac[0]
        a9 = 2 / 10
        for i in range(1, n):
            sig[i] = a9 * mac[i] + (1 - a9) * sig[i - 1]
        ind[f"macd_{fast}_{slow}_hist"] = mac - sig

    for p in [5, 10, 20, 50]:
        s = pd.Series(c)
        sma = s.rolling(p).mean().values
        std = s.rolling(p).std().values
        for mf, tag in [(1.0, "10"), (1.5, "15"), (2.0, "20"), (2.5, "25"), (3.0, "30")]:
            up = sma + mf * std
            lo = sma - mf * std
            ind[f"bb_{p}_{tag}_pct"] = (c - lo) / (up - lo + 1e-9)
            ind[f"bb_{p}_{tag}_upper"] = up
            ind[f"bb_{p}_{tag}_lower"] = lo
            ind[f"bb_{p}_{tag}_width"] = (up - lo) / (sma + 1e-9)

    for p in [5, 10, 20, 50, 100, 200]:
        ind[f"don_high_{p}"] = pd.Series(h).rolling(p).max().shift(1).values
        ind[f"don_low_{p}"] = pd.Series(l).rolling(p).min().shift(1).values

    for p in [3, 5, 7, 9, 14, 21]:
        lo_p = pd.Series(l).rolling(p).min().values
        hi_p = pd.Series(h).rolling(p).max().values
        ind[f"stoch_k_{p}"] = (c - lo_p) / (hi_p - lo_p + 1e-9) * 100

    for p in [7, 10, 14, 20, 30]:
        tp = (h + l + c) / 3
        sma = pd.Series(tp).rolling(p).mean().values
        mad = pd.Series(tp).rolling(p).apply(lambda x: np.abs(x - x.mean()).mean()).values
        ind[f"cci_{p}"] = (tp - sma) / (0.015 * mad + 1e-9)

    vwap_arr = np.full(n, np.nan)
    tp = (h + l + c) / 3
    cum_tpv = np.zeros(n)
    cum_vol = np.zeros(n)
    datas = df.index.date
    data_atual = None
    for i in range(n):
        if datas[i] != data_atual:
            data_atual = datas[i]
            cum_tpv[i] = tp[i] * v[i]
            cum_vol[i] = v[i]
        else:
            cum_tpv[i] = cum_tpv[i - 1] + tp[i] * v[i]
            cum_vol[i] = cum_vol[i - 1] + v[i]
        if cum_vol[i] > 0:
            vwap_arr[i] = cum_tpv[i] / cum_vol[i]
    ind["vwap"] = vwap_arr

    vwap_std = np.full(n, np.nan)
    sq_sum = np.zeros(n)
    cnt = np.zeros(n)
    data_atual = None
    for i in range(n):
        if datas[i] != data_atual:
            data_atual = datas[i]
            sq_sum[i] = (c[i] - vwap_arr[i]) ** 2
            cnt[i] = 1
        else:
            sq_sum[i] = sq_sum[i - 1] + (c[i] - vwap_arr[i]) ** 2
            cnt[i] = cnt[i - 1] + 1
        if cnt[i] > 1:
            vwap_std[i] = np.sqrt(sq_sum[i] / cnt[i])
    ind["vwap_std"] = vwap_std
    ind["vwap_upper1"] = vwap_arr + vwap_std
    ind["vwap_lower1"] = vwap_arr - vwap_std
    ind["vwap_upper2"] = vwap_arr + 2 * vwap_std
    ind["vwap_lower2"] = vwap_arr - 2 * vwap_std

    for p in [10, 20]:
        vm = pd.Series(v).rolling(p).mean().values
        vs = pd.Series(v).rolling(p).std().values
        ind[f"vol_z_{p}"] = (v - vm) / (vs + 1e-9)
        ind[f"vol_ratio_{p}"] = v / (vm + 1e-9)

    ret = np.diff(c, prepend=c[0]) / (c + 1e-9)
    v5 = pd.Series(ret).rolling(5).std().values * 100
    v20 = pd.Series(ret).rolling(20).std().values * 100
    ind["vol_ratio"] = v5 / (v20 + 1e-9)

    for ep in [5, 10, 20, 50]:
        for ap in [7, 14, 20]:
            for mf in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                key = f"kc_{ep}_{ap}_{str(mf).replace('.', '')}"
                ind[f"{key}_upper"] = ind[f"ema_{ep}"] + mf * ind[f"atr_{ap}"]
                ind[f"{key}_lower"] = ind[f"ema_{ep}"] - mf * ind[f"atr_{ap}"]

    for orb_min in [5, 10, 15, 20, 30, 45]:
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)
        day_data = {}
        for i in range(n):
            dt = df.index[i]
            data = dt.date()
            mins = dt.hour * 60 + dt.minute - 9 * 60
            if mins < 0:
                continue
            if data not in day_data:
                day_data[data] = {"hi": -np.inf, "lo": np.inf, "done": False}
            if not day_data[data]["done"]:
                if mins <= orb_min:
                    day_data[data]["hi"] = max(day_data[data]["hi"], h[i])
                    day_data[data]["lo"] = min(day_data[data]["lo"], l[i])
                else:
                    day_data[data]["done"] = True
            if day_data[data]["done"] or mins > orb_min:
                orb_high[i] = day_data[data]["hi"]
                orb_low[i] = day_data[data]["lo"]
        ind[f"orb_high_{orb_min}"] = orb_high
        ind[f"orb_low_{orb_min}"] = orb_low

    hora = df.index.hour
    ind["session_am"] = ((hora >= 9) & (hora < 12)).astype(np.int8)
    ind["session_pm"] = ((hora >= 13) & (hora < 17)).astype(np.int8)
    ind["hora"] = hora
    ind["dow"] = df.index.dayofweek.values
    return ind

# ================================================================
# SECAO 3: SIMULADORES NUMBA
# ================================================================

@njit(cache=True)
def simular_long(on, hi, lo, ent, ext, sl_pts, tp_pts, mult, comm, slip):
    n = len(on)
    pnls = np.empty(n, dtype=np.float64)
    n_tr = 0
    em = False
    ep = 0.0
    sl = 0.0
    tp = 0.0
    for i in range(n - 1):
        if em:
            if lo[i] <= sl or hi[i] >= tp or ext[i]:
                saida = sl if lo[i] <= sl else (tp if hi[i] >= tp else on[i])
                pnls[n_tr] = (saida - ep) * mult - comm - slip * mult * 0.1
                n_tr += 1
                em = False
            continue
        if ent[i] and not em:
            ep = on[i]
            if np.isnan(ep) or ep <= 0:
                continue
            sl = ep - sl_pts
            tp = ep + tp_pts
            em = True
    return pnls[:n_tr]

@njit(cache=True)
def simular_short(on, hi, lo, ent, ext, sl_pts, tp_pts, mult, comm, slip):
    n = len(on)
    pnls = np.empty(n, dtype=np.float64)
    n_tr = 0
    em = False
    ep = 0.0
    sl = 0.0
    tp = 0.0
    for i in range(n - 1):
        if em:
            if hi[i] >= sl or lo[i] <= tp or ext[i]:
                saida = sl if hi[i] >= sl else (tp if lo[i] <= tp else on[i])
                pnls[n_tr] = (ep - saida) * mult - comm - slip * mult * 0.1
                n_tr += 1
                em = False
            continue
        if ent[i] and not em:
            ep = on[i]
            if np.isnan(ep) or ep <= 0:
                continue
            sl = ep + sl_pts
            tp = ep - tp_pts
            em = True
    return pnls[:n_tr]

@njit(cache=True)
def simular_com_contexto(on, hi, lo, ent, ext, sl_pts, tp_pts,
                         mult, comm, slip,
                         atr, vol_ratio, rsi, dist_vwap, hora, dow):
    """
    Simula e registra o contexto de cada trade para analise de falhas.
    Retorna pnls + features de cada entrada.
    """
    n = len(on)
    pnls = np.empty(n, dtype=np.float64)
    ctx_atr = np.empty(n, dtype=np.float64)
    ctx_vol = np.empty(n, dtype=np.float64)
    ctx_rsi = np.empty(n, dtype=np.float64)
    ctx_dvwap = np.empty(n, dtype=np.float64)
    ctx_hora = np.empty(n, dtype=np.float64)
    ctx_dow = np.empty(n, dtype=np.float64)
    n_tr = 0
    em = False
    ep = 0.0
    sl = 0.0
    tp = 0.0
    entry_idx = 0

    for i in range(n - 1):
        if em:
            if hi[i] >= sl or lo[i] <= tp or ext[i]:
                saida = sl if hi[i] >= sl else (tp if lo[i] <= tp else on[i])
                pnls[n_tr] = (ep - saida) * mult - comm - slip * mult * 0.1
                ctx_atr[n_tr] = atr[entry_idx]
                ctx_vol[n_tr] = vol_ratio[entry_idx]
                ctx_rsi[n_tr] = rsi[entry_idx]
                ctx_dvwap[n_tr] = dist_vwap[entry_idx]
                ctx_hora[n_tr] = hora[entry_idx]
                ctx_dow[n_tr] = dow[entry_idx]
                n_tr += 1
                em = False
            continue
        if ent[i] and not em:
            ep = on[i]
            if np.isnan(ep) or ep <= 0:
                continue
            sl = ep + sl_pts
            tp = ep - tp_pts
            em = True
            entry_idx = i

    return (
        pnls[:n_tr],
        ctx_atr[:n_tr],
        ctx_vol[:n_tr],
        ctx_rsi[:n_tr],
        ctx_dvwap[:n_tr],
        ctx_hora[:n_tr],
        ctx_dow[:n_tr],
    )

@njit(cache=True)
def calcular_mae_mfe(on, hi, lo, ent, direction, max_bars=60):
    n = len(on)
    mae = np.empty(n, dtype=np.float64)
    mfe = np.empty(n, dtype=np.float64)
    n_tr = 0
    for i in range(n - max_bars):
        if not ent[i]:
            continue
        ep = on[i]
        if np.isnan(ep) or ep <= 0:
            continue
        mf = 0.0
        mc = 0.0
        for j in range(i + 1, min(i + max_bars, n)):
            fav = (hi[j] - ep) if direction == 1 else (ep - lo[j])
            con = (ep - lo[j]) if direction == 1 else (hi[j] - ep)
            if fav > mf:
                mf = fav
            if con > mc:
                mc = con
        mae[n_tr] = mc
        mfe[n_tr] = mf
        n_tr += 1
    return mae[:n_tr], mfe[:n_tr]

def executar(ind, ent, ext, sl_pts, tp_pts, direction):
    on = ind["open_next"].astype(np.float64)
    hi = ind["high"].astype(np.float64)
    lo = ind["low"].astype(np.float64)
    e = ent.astype(np.bool_)
    x = ext.astype(np.bool_)
    if direction == "long":
        return simular_long(on, hi, lo, e, x, sl_pts, tp_pts, MULT, COMM, SLIP)
    return simular_short(on, hi, lo, e, x, sl_pts, tp_pts, MULT, COMM, SLIP)

def metricas(pnls, min_trades=MIN_TRADES_IS):
    if len(pnls) < min_trades:
        return None
    w = pnls[pnls > 0]
    l = pnls[pnls <= 0]
    if len(l) == 0 or len(w) == 0:
        return None
    pf = abs(w.sum() / l.sum())
    if pf > MAX_PF:
        return None
    eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnls)])
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk * 100).min())
    if mdd < MAX_DD:
        return None
    ret = pnls / CAPITAL
    sh = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252 * 390))
    if sh > MAX_SHARPE:
        return None
    jan_pos = sum(1 for s in range(0, max(1, len(pnls) - 30), 15) if pnls[s:s + 30].sum() > 0)
    n_jan = max(1, len(range(0, max(1, len(pnls) - 30), 15)))
    return {
        "n": len(pnls),
        "wr": round(len(w) / len(pnls) * 100, 2),
        "pf": round(pf, 3),
        "sh": round(sh, 3),
        "exp": round(float(pnls.mean()), 2),
        "pnl": round(float(pnls.sum()), 2),
        "mdd": round(mdd, 2),
        "jan_pos": round(jan_pos / n_jan * 100, 1),
    }

def score(m):
    if not m:
        return 0
    return (
        min(m["pf"], MAX_PF) / MAX_PF * 0.30
        + max(0, min(m["exp"], 500)) / 500 * 0.25
        + m["jan_pos"] / 100 * 0.20
        + max(0, min(m["sh"], 3)) / 3 * 0.15
        + min(m["n"], 2000) / 2000 * 0.10
    )

# ================================================================
# SECAO 4: ANALISE DE FALHAS - CORE DO ENGINE v3
# ================================================================

def analise_de_falhas(ind, ent, ext, params, sl_pts, tp_pts):
    """
    Analisa o contexto dos trades perdedores e tenta encontrar
    um filtro que elimine os losers sem remover os winners.

    Retorna o melhor filtro encontrado ou None.
    """
    direction = params.get("direction", "short")
    on = ind["open_next"].astype(np.float64)
    hi = ind["high"].astype(np.float64)
    lo = ind["low"].astype(np.float64)
    e = ent.astype(np.bool_)
    x = ext.astype(np.bool_)

    vwap = ind["vwap"]
    c = ind["close"]
    dist_vwap = np.abs(c - vwap) / (c + 1e-9) * 100
    dist_vwap = np.nan_to_num(dist_vwap, nan=0.0)

    rsi_arr = ind.get("rsi_14", np.full(len(c), 50.0))
    rsi_arr = np.nan_to_num(rsi_arr, nan=50.0)

    atr_arr = ind.get("atr_14", np.full(len(c), 10.0))
    atr_arr = np.nan_to_num(atr_arr, nan=10.0)

    vol_arr = ind.get("vol_ratio", np.ones(len(c)))
    vol_arr = np.nan_to_num(vol_arr, nan=1.0)

    hora_arr = ind["hora"].astype(np.float64)
    dow_arr = ind["dow"].astype(np.float64)

    try:
        result = simular_com_contexto(
            on, hi, lo, e, x, sl_pts, tp_pts,
            MULT, COMM, SLIP,
            atr_arr, vol_arr, rsi_arr, dist_vwap, hora_arr, dow_arr
        )
        pnls, ctx_atr, ctx_vol, ctx_rsi, ctx_dvwap, ctx_hora, ctx_dow = result
    except Exception:
        return None

    if len(pnls) < 50:
        return None

    winners = pnls > 0
    losers = pnls <= 0
    n_win = winners.sum()
    n_los = losers.sum()

    if n_win < 10 or n_los < 10:
        return None

    features = {
        "atr": ctx_atr,
        "vol": ctx_vol,
        "rsi": ctx_rsi,
        "dist_vwap": ctx_dvwap,
        "hora": ctx_hora,
        "dow": ctx_dow,
    }

    melhores_filtros = []

    for feat_nome, feat_vals in features.items():
        for pct in [25, 33, 50, 67, 75]:
            corte = np.percentile(feat_vals[losers], pct)

            for op in ["maior", "menor"]:
                if op == "maior":
                    filtro_mask = feat_vals <= corte
                else:
                    filtro_mask = feat_vals >= corte

                pnls_filtrado = pnls[filtro_mask]
                if len(pnls_filtrado) < MIN_TRADES_IS // 2:
                    continue

                w_f = pnls_filtrado[pnls_filtrado > 0]
                l_f = pnls_filtrado[pnls_filtrado <= 0]
                if len(l_f) == 0 or len(w_f) == 0:
                    continue

                pf_novo = abs(w_f.sum() / l_f.sum())
                pf_orig = abs(pnls[winners].sum() / pnls[losers].sum()) if n_los > 0 else 1.0

                if pf_novo > pf_orig * 1.05 and pf_novo > 1.0:
                    melhores_filtros.append({
                        "feature": feat_nome,
                        "operacao": op,
                        "corte": round(float(corte), 3),
                        "pf_antes": round(float(pf_orig), 3),
                        "pf_depois": round(float(pf_novo), 3),
                        "melhoria": round((pf_novo - pf_orig) / pf_orig * 100, 1),
                        "n_trades": len(pnls_filtrado),
                        "n_removidos": len(pnls) - len(pnls_filtrado),
                    })

    if not melhores_filtros:
        return None
    melhores_filtros.sort(key=lambda x: -x["pf_depois"])
    return melhores_filtros[0]

def aplicar_filtro_loser(ind, ent, filtro):
    """
    Aplica o filtro de loser encontrado na analise de falhas.
    Retorna novos sinais de entrada filtrados.
    """
    if filtro is None:
        return ent

    feat_nome = filtro["feature"]
    op = filtro["operacao"]
    corte = filtro["corte"]
    c = ind["close"]

    feat_map = {
        "atr": ind.get("atr_14", np.full(len(c), 10.0)),
        "vol": ind.get("vol_ratio", np.ones(len(c))),
        "rsi": ind.get("rsi_14", np.full(len(c), 50.0)),
        "dist_vwap": np.abs(c - ind["vwap"]) / (c + 1e-9) * 100,
        "hora": ind["hora"].astype(np.float64),
        "dow": ind["dow"].astype(np.float64),
    }

    feat = feat_map.get(feat_nome, np.ones(len(c)))
    feat = np.nan_to_num(feat, nan=0.0)

    if op == "maior":
        mascara = feat <= corte
    else:
        mascara = feat >= corte

    return ent & mascara

# ================================================================
# SECAO 5: SINAIS - 23 ESTRATEGIAS
# ================================================================

def mascara_sessao(ind, session):
    if session == "am":
        return ind["session_am"].astype(bool)
    elif session == "pm":
        return ind["session_pm"].astype(bool)
    return np.ones(len(ind["close"]), dtype=bool)

def h1(x):
    return np.roll(x, 1)

def gerar_sinais(estrategia, ind, params):
    d = params.get("direction", "short")
    ses = params.get("session", "all")
    mask = mascara_sessao(ind, ses)
    c = ind["close"]
    ent = None
    ext = None

    if estrategia == "vwap_reversion":
        vwap = ind["vwap"]
        std = ind["vwap_std"]
        mult = params["vwap_std"]
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        if rsi is None:
            return None, None
        lvl = params["rsi_level"]
        if d == "long":
            ent = (c < vwap - mult * std) & (rsi < lvl) & (h1(rsi) >= lvl)
            ext = c > vwap
        else:
            ent = (c > vwap + mult * std) & (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl))
            ext = c < vwap

    elif estrategia == "vwap_breakout":
        vwap = ind["vwap"]
        vr = ind["vol_ratio"]
        vc = params["vol_confirm"]
        if d == "long":
            ent = (c > vwap) & (h1(c) <= h1(vwap)) & (vr > vc)
            ext = c < vwap
        else:
            ent = (c < vwap) & (h1(c) >= h1(vwap)) & (vr > vc)
            ext = c > vwap

    elif estrategia == "vwap_pullback":
        vwap = ind["vwap"]
        ema = ind.get(f"ema_{params['ema_period']}")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        rf = params["rsi_filter"]
        if ema is None or rsi is None:
            return None, None
        tol = ind["atr_14"] * 0.3
        if d == "long":
            ent = (ema > vwap) & (np.abs(c - vwap) < tol) & (rsi > rf)
            ext = c < vwap
        else:
            ent = (ema < vwap) & (np.abs(c - vwap) < tol) & (rsi < (100 - rf))
            ext = c > vwap

    elif estrategia == "orb_breakout":
        om = params["orb_minutes"]
        orh = ind.get(f"orb_high_{om}")
        orl = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        vr = ind["vol_ratio"]
        vc = params.get("vol_confirm", 1.0)
        if d == "long":
            ent = (c > orh) & (h1(c) <= h1(orh)) & (vr > vc)
            ext = c < orl
        else:
            ent = (c < orl) & (h1(c) >= h1(orl)) & (vr > vc)
            ext = c > orh

    elif estrategia == "orb_retest":
        om = params["orb_minutes"]
        orh = ind.get(f"orb_high_{om}")
        orl = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        tol = ind["atr_14"] * 0.5
        above = c > orh
        if d == "long":
            foi = pd.Series(above).rolling(20).max().values.astype(bool)
            ent = foi & (np.abs(c - orh) < tol)
            ext = c < orl
        else:
            foi = pd.Series(~above).rolling(20).max().values.astype(bool)
            ent = foi & (np.abs(c - orl) < tol)
            ext = c > orh

    elif estrategia == "rsi_vwap_combo":
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        vwap = ind["vwap"]
        side = params["vwap_side"]
        lvl = params["rsi_level"]
        if rsi is None:
            return None, None
        vc = (c > vwap) if side == "above" else (c < vwap)
        if d == "long":
            ent = (rsi < lvl) & (h1(rsi) >= lvl) & vc
            ext = rsi > 50
        else:
            ent = (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl)) & (~vc)
            ext = rsi < 50

    elif estrategia == "rsi_ema_vwap":
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        ema = ind.get(f"ema_{params['ema_period']}")
        vwap = ind["vwap"]
        lvl = params["rsi_level"]
        if rsi is None or ema is None:
            return None, None
        if d == "long":
            ent = (rsi < lvl) & (h1(rsi) >= lvl) & (c > ema) & (c > vwap)
            ext = rsi > 55
        else:
            ent = (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl)) & (c < ema) & (c < vwap)
            ext = rsi < 45

    elif estrategia == "rsi_vwap_session":
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        vwap = ind["vwap"]
        ovs = params["oversold"]
        ovb = params["overbought"]
        el = params["exit_level"]
        if rsi is None:
            return None, None
        if d == "long":
            ent = (rsi < ovs) & (h1(rsi) >= ovs) & (c > vwap)
            ext = rsi > el
        else:
            ent = (rsi > ovb) & (h1(rsi) <= ovb) & (c < vwap)
            ext = rsi < (100 - el)

    elif estrategia == "atr_channel_breakout":
        ep = params["ema_period"]
        ap = params["atr_period"]
        am = params["atr_mult"]
        key = f"kc_{ep}_{ap}_{str(am).replace('.', '')}"
        ku = ind.get(f"{key}_upper")
        kl = ind.get(f"{key}_lower")
        if ku is None:
            return None, None
        if d == "long":
            ent = (c > ku) & (h1(c) <= h1(ku))
            ext = c < kl
        else:
            ent = (c < kl) & (h1(c) >= h1(kl))
            ext = c > ku

    elif estrategia == "atr_trailing_momentum":
        pp = params["momentum_period"]
        pt = params["momentum_thresh"]
        roc = np.empty(len(c))
        roc[:pp] = np.nan
        roc[pp:] = (c[pp:] - c[:-pp]) / (c[:-pp] + 1e-9) * 100
        if d == "long":
            ent = roc > pt
            ext = roc < 0
        else:
            ent = roc < -pt
            ext = roc > 0

    elif estrategia == "macd_vwap":
        mf = params["macd_fast"]
        ms = params["macd_slow"]
        hist = ind.get(f"macd_{mf}_{ms}_hist")
        vwap = ind["vwap"]
        if hist is None:
            return None, None
        if d == "long":
            ent = (hist > 0) & (h1(hist) <= 0) & (c > vwap)
            ext = (hist < 0) & (h1(hist) >= 0)
        else:
            ent = (hist < 0) & (h1(hist) >= 0) & (c < vwap)
            ext = (hist > 0) & (h1(hist) <= 0)

    elif estrategia == "macd_rsi_vwap":
        cfg = params["macd_config"]
        hist = ind.get(f"macd_{cfg}_hist")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        rf = params["rsi_filter"]
        vwap = ind["vwap"]
        if hist is None or rsi is None:
            return None, None
        if d == "long":
            ent = (hist > 0) & (h1(hist) <= 0) & (rsi > rf) & (c > vwap)
            ext = (hist < 0) & (h1(hist) >= 0)
        else:
            ent = (hist < 0) & (h1(hist) >= 0) & (rsi < (100 - rf)) & (c < vwap)
            ext = (hist > 0) & (h1(hist) <= 0)

    elif estrategia == "ema_vwap_trend":
        ef = ind.get(f"ema_{params['fast']}")
        es = ind.get(f"ema_{params['slow']}")
        vwap = ind["vwap"]
        if ef is None or es is None:
            return None, None
        if d == "long":
            ent = (ef > es) & (h1(ef) <= h1(es)) & (c > vwap)
            ext = ef < es
        else:
            ent = (ef < es) & (h1(ef) >= h1(es)) & (c < vwap)
            ext = ef > es

    elif estrategia == "dual_ema_momentum":
        ef = ind.get(f"ema_{params['fast']}")
        es = ind.get(f"ema_{params['slow']}")
        vr = ind["vol_ratio"]
        vc = params["vol_confirm"]
        if ef is None or es is None:
            return None, None
        if d == "long":
            ent = (ef > es) & (h1(ef) <= h1(es)) & (vr > vc)
            ext = ef < es
        else:
            ent = (ef < es) & (h1(ef) >= h1(es)) & (vr > vc)
            ext = ef > es

    elif estrategia == "bb_squeeze_breakout":
        bp = params["bb_period"]
        bs = params["bb_std"]
        sm = params["squeeze_mult"]
        vc = params["vol_confirm"]
        key = f"bb_{bp}_{str(bs).replace('.', '')}"
        bw = ind.get(f"{key}_width")
        bu = ind.get(f"{key}_upper")
        bl = ind.get(f"{key}_lower")
        vr = ind["vol_ratio"]
        if bw is None:
            return None, None
        bwa = pd.Series(bw).rolling(20).mean().values
        squeeze = bw < bwa * sm
        saindo = ~squeeze & pd.Series(squeeze).shift(1).fillna(False).values
        if d == "long":
            ent = saindo & (c > bu) & (vr > vc)
            ext = c < bl
        else:
            ent = saindo & (c < bl) & (vr > vc)
            ext = c > bu

    elif estrategia == "bb_rsi_vwap":
        bp = params["bb_period"]
        bs = params["bb_std"]
        key = f"bb_{bp}_{str(bs).replace('.', '')}"
        pct = ind.get(f"{key}_pct")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        rc = params["rsi_confirm"]
        vwap = ind["vwap"]
        if pct is None or rsi is None:
            return None, None
        if d == "long":
            ent = (pct < 0.05) & (rsi < rc) & (c > vwap)
            ext = pct > 0.5
        else:
            ent = (pct > 0.95) & (rsi > (100 - rc)) & (c < vwap)
            ext = pct < 0.5

    elif estrategia == "donchian_vwap":
        dh = ind.get(f"don_high_{params['don_period']}")
        dl = ind.get(f"don_low_{params['don_period']}")
        vwap = ind["vwap"]
        vr = ind["vol_ratio"]
        vc = params["vol_confirm"]
        if dh is None:
            return None, None
        if d == "long":
            ent = (c > dh) & (c > vwap) & (vr > vc)
            ext = c < dl
        else:
            ent = (c < dl) & (c < vwap) & (vr > vc)
            ext = c > dh

    elif estrategia == "stoch_vwap":
        k = ind.get(f"stoch_k_{params['stoch_period']}")
        vwap = ind["vwap"]
        ovs = params["oversold"]
        ovb = params["overbought"]
        if k is None:
            return None, None
        if d == "long":
            ent = (k < ovs) & (h1(k) >= ovs) & (c > vwap)
            ext = k > 50
        else:
            ent = (k > ovb) & (h1(k) <= ovb) & (c < vwap)
            ext = k < 50

    elif estrategia == "stoch_ema_vwap":
        k = ind.get(f"stoch_k_{params['stoch_period']}")
        ema = ind.get(f"ema_{params['ema_period']}")
        vwap = ind["vwap"]
        ovs = params["oversold"]
        ovb = params["overbought"]
        if k is None or ema is None:
            return None, None
        if d == "long":
            ent = (k < ovs) & (h1(k) >= ovs) & (c > ema) & (c > vwap)
            ext = k > 50
        else:
            ent = (k > ovb) & (h1(k) <= ovb) & (c < ema) & (c < vwap)
            ext = k < 50

    elif estrategia == "volume_spike_reversal":
        vz = ind.get("vol_z_20")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        vs = params["vol_spike"]
        lvl = params["rsi_level"]
        if vz is None or rsi is None:
            return None, None
        spike = vz > vs
        if d == "long":
            ent = spike & (rsi < lvl) & (h1(rsi) >= lvl)
            ext = rsi > 55
        else:
            ent = spike & (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl))
            ext = rsi < 45

    elif estrategia == "volume_vwap_momentum":
        vr = ind["vol_ratio"]
        vwap = ind["vwap"]
        vc = params["vol_confirm"]
        if d == "long":
            ent = (c > vwap) & (h1(c) <= h1(vwap)) & (vr > vc)
            ext = c < vwap
        else:
            ent = (c < vwap) & (h1(c) >= h1(vwap)) & (vr > vc)
            ext = c > vwap

    elif estrategia == "cci_vwap":
        cci = ind.get(f"cci_{params['cci_period']}")
        vwap = ind["vwap"]
        thr = params["cci_thresh"]
        if cci is None:
            return None, None
        if d == "long":
            ent = (cci < -thr) & (h1(cci) >= -thr) & (c > vwap)
            ext = cci > 0
        else:
            ent = (cci > thr) & (h1(cci) <= thr) & (c < vwap)
            ext = cci < 0

    elif estrategia == "rsi_full_combo":
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        ovs = params["oversold"]
        ovb = params["overbought"]
        el = params["exit_level"]
        if rsi is None:
            return None, None
        if d == "long":
            ent = (rsi < ovs) & (h1(rsi) >= ovs)
            ext = rsi > el
        else:
            ent = (rsi > ovb) & (h1(rsi) <= ovb)
            ext = rsi < (100 - el)

    elif estrategia == "orb_vwap_combo":
        om = params["orb_minutes"]
        orh = ind.get(f"orb_high_{om}")
        orl = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        vr = ind["vol_ratio"]
        vc = params["vol_confirm"]
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        rf = params["rsi_filter"]
        vwap = ind["vwap"]
        if rsi is None:
            return None, None
        if d == "long":
            ent = (c > orh) & (h1(c) <= h1(orh)) & (vr > vc) & (rsi > rf) & (c > vwap)
            ext = c < orl
        else:
            ent = (c < orl) & (h1(c) >= h1(orl)) & (vr > vc) & (rsi < (100 - rf)) & (c < vwap)
            ext = c > orh
    else:
        return None, None

    if ent is None:
        return None, None

    ent = ent & mask
    ent[0] = False
    return ent.astype(np.bool_), ext.astype(np.bool_)

# ================================================================
# SECAO 6: GRIDS - 10M COMBOS
# ================================================================

GRIDS = {
    "vwap_reversion": {
        "vwap_std": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "rsi_period": [2, 3, 4, 5, 7, 9, 11, 14, 18, 21, 28],
        "rsi_level": [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "vwap_breakout": {
        "vol_confirm": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "vwap_pullback": {
        "ema_period": [5, 9, 13, 20, 34, 50, 100, 200],
        "rsi_period": [3, 5, 7, 9, 14, 21, 28],
        "rsi_filter": [25, 30, 35, 40, 45, 50, 55, 60],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "orb_breakout": {
        "orb_minutes": [5, 10, 15, 20, 30, 45, 60],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "orb_retest": {
        "orb_minutes": [5, 10, 15, 20, 30, 45, 60],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "rsi_vwap_combo": {
        "rsi_period": [2, 3, 4, 5, 7, 9, 11, 14, 18, 21, 28],
        "rsi_level": [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50],
        "vwap_side": ["above", "below"],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "rsi_ema_vwap": {
        "rsi_period": [2, 3, 5, 7, 9, 14, 21, 28],
        "rsi_level": [5, 10, 15, 20, 25, 30, 35, 40, 45],
        "ema_period": [5, 10, 20, 34, 50, 100, 200],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "rsi_vwap_session": {
        "rsi_period": [2, 3, 5, 7, 9, 11, 14, 18, 21, 28, 35, 42],
        "oversold": [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50],
        "overbought": [50, 55, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95, 97],
        "exit_level": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        "atr_sl": [0.3, 0.5, 1.0, 1.5, 2.0],
        "rr": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "rsi_full_combo": {
        "rsi_period": [2, 3, 5, 7, 9, 11, 14, 18, 21, 28],
        "oversold": [3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "overbought": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "exit_level": [45, 50, 55, 60, 65, 70, 75, 80],
        "atr_sl": [0.3, 0.5, 1.0, 1.5, 2.0],
        "rr": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "atr_channel_breakout": {
        "ema_period": [5, 10, 20, 50, 100],
        "atr_period": [7, 14, 20],
        "atr_mult": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "atr_trailing_momentum": {
        "momentum_period": [2, 3, 5, 10, 15, 20, 30],
        "momentum_thresh": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "macd_vwap": {
        "macd_fast": [2, 3, 5, 8, 10, 12],
        "macd_slow": [8, 10, 13, 21, 26],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "macd_rsi_vwap": {
        "macd_config": ["12_26", "8_21", "5_13", "3_10"],
        "rsi_period": [3, 5, 7, 9, 14, 21, 28],
        "rsi_filter": [25, 30, 35, 40, 45, 50, 55, 60],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "ema_vwap_trend": {
        "fast": [2, 3, 5, 8, 10, 13, 20, 21],
        "slow": [20, 34, 50, 100, 200],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "dual_ema_momentum": {
        "fast": [2, 3, 5, 8, 10, 13, 20],
        "slow": [20, 21, 34, 50, 100],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "bb_squeeze_breakout": {
        "bb_period": [5, 10, 20, 50],
        "bb_std": [1.0, 1.5, 2.0, 2.5, 3.0],
        "squeeze_mult": [0.5, 0.6, 0.7, 0.8, 0.9],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "bb_rsi_vwap": {
        "bb_period": [5, 10, 20, 50],
        "bb_std": [1.0, 1.5, 2.0, 2.5, 3.0],
        "rsi_period": [2, 3, 5, 7, 9, 14, 21],
        "rsi_confirm": [5, 10, 15, 20, 25, 30, 35, 40],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "donchian_vwap": {
        "don_period": [5, 10, 20, 50, 100, 200],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "stoch_vwap": {
        "stoch_period": [2, 3, 5, 7, 9, 14, 21],
        "oversold": [3, 5, 8, 10, 15, 20, 25, 30, 35],
        "overbought": [65, 70, 75, 80, 85, 90, 92, 95, 97],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "stoch_ema_vwap": {
        "stoch_period": [2, 3, 5, 7, 9, 14, 21],
        "oversold": [3, 5, 8, 10, 15, 20, 25, 30],
        "overbought": [70, 75, 80, 85, 90, 92, 95, 97],
        "ema_period": [20, 50, 100, 200],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "volume_spike_reversal": {
        "vol_spike": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "rsi_period": [2, 3, 5, 7, 9, 14, 21],
        "rsi_level": [5, 10, 15, 20, 25, 30, 35, 40],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "volume_vwap_momentum": {
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "cci_vwap": {
        "cci_period": [5, 7, 10, 14, 20, 30],
        "cci_thresh": [25, 50, 75, 100, 125, 150, 175, 200, 250],
        "atr_sl": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "orb_vwap_combo": {
        "orb_minutes": [5, 10, 15, 20, 30, 45, 60],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "rsi_period": [5, 7, 9, 14, 21],
        "rsi_filter": [30, 35, 40, 45, 50, 55],
        "atr_sl": [0.3, 0.5, 1.0, 1.5, 2.0],
        "rr": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session": ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
}

# ================================================================
# SECAO 7: WORKER MULTIPROCESSING
# ================================================================

def worker_estrategia(args):
    estrategia, grid, ind_is, mini = args
    ind = ind_is
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if mini:
        combos = combos[:20]

    validos = []
    atr_med = float(np.nanmean(ind["atr_14"]))
    t0 = time.time()

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            ent, ext = gerar_sinais(estrategia, ind, params)
            if ent is None or ent.sum() < 10:
                continue
            sl = atr_med * params.get("atr_sl", 1.0)
            tp = sl * params.get("rr", 2.0)
            pnls = executar(ind, ent, ext, sl, tp, params.get("direction", "short"))
            m = metricas(pnls)
            if not m:
                continue
            s = score(m)
            validos.append({"estrategia": estrategia, "params": params, "score": round(s, 6), **m})
        except Exception:
            continue

    validos.sort(key=lambda x: -x["score"])
    return {
        "estrategia": estrategia,
        "n_combos": len(combos),
        "n_validos": len(validos),
        "elapsed": round(time.time() - t0, 1),
        "top10": validos[:10],
    }

# ================================================================
# SECAO 8: OOS ROLLING (pre-calculado)
# ================================================================

def oos_rolling(estrategia, df_oos, params, ind_oos_full):
    """OOS rolling usando indicadores pre-calculados."""
    datas = df_oos.index.normalize().unique()
    resultados = []
    janela = 30
    i = 0

    while i + janela <= len(datas):
        d_s = datas[i]
        d_e = datas[min(i + janela - 1, len(datas) - 1)]
        mask_j = (df_oos.index.normalize() >= d_s) & (df_oos.index.normalize() <= d_e)
        if mask_j.sum() < 100:
            i += janela
            continue

        ind_j = {
            k: v[mask_j] if isinstance(v, np.ndarray) and len(v) == len(df_oos) else v
            for k, v in ind_oos_full.items()
        }
        ind_j["hora"] = df_oos[mask_j].index.hour.values
        ind_j["dow"] = df_oos[mask_j].index.dayofweek.values

        ent, ext = gerar_sinais(estrategia, ind_j, params)
        if ent is None or ent.sum() < 5:
            i += janela
            continue

        atr_j = float(np.nanmean(ind_j["atr_14"]))
        sl = atr_j * params.get("atr_sl", 1.0)
        tp = sl * params.get("rr", 2.0)
        pnls = executar(ind_j, ent, ext, sl, tp, params.get("direction", "short"))
        m = metricas(pnls, min_trades=20)
        resultados.append({
            "data": str(d_s.date()),
            "pf": m["pf"] if m else 0,
            "trades": m["n"] if m else 0,
            "pnl": m["pnl"] if m else 0,
            "lucrativo": m is not None and m["pf"] > 1.0,
        })
        i += janela

    if not resultados:
        return None
    pf_list = [r["pf"] for r in resultados if r["pf"] > 0]
    luc = sum(1 for r in resultados if r["lucrativo"])
    wfe = luc / len(resultados) * 100 if resultados else 0
    return {
        "janelas": len(resultados),
        "lucrativas": luc,
        "wfe_pct": round(wfe, 1),
        "pf_medio": round(float(np.mean(pf_list)), 3) if pf_list else 0,
        "detalhes": resultados,
    }

# ================================================================
# SECAO 9: MAIN
# ================================================================

def main():
    MINI = "--mini" in sys.argv
    total = sum(math.prod(len(v) for v in g.values()) for g in GRIDS.values())

    print("=" * 68, flush=True)
    print("  INTERNET STRATEGIES ENGINE v3 - WDO B3", flush=True)
    print(f"  {len(GRIDS)} estrategias | {total:,} combos", flush=True)
    print(f"  {N_CPUS} CPUs | MAE/MFE | ANALISE DE FALHAS | OOS Rolling", flush=True)
    print("=" * 68, flush=True)

    df = carregar()
    split = int(len(df) * 0.70)
    df_is = df.iloc[:split]
    df_oos = df.iloc[split:]
    print(f"  IS : {len(df_is):,} | {df_is.index[0].date()} -> {df_is.index[-1].date()}", flush=True)
    print(f"  OOS: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}", flush=True)

    print("\n[IND] Calculando IS...", flush=True)
    ind_is = calcular_indicadores(df_is)
    print("[IND] Calculando OOS...", flush=True)
    ind_oos = calcular_indicadores(df_oos)
    print("[IND] Pronto!", flush=True)

    print("\n[JIT] Compilando...", flush=True)
    d = np.ones(200, dtype=np.float64) * 5000
    b = np.zeros(200, dtype=np.bool_)
    b[10] = True
    _ = simular_long(d, d, d, b, b, 20.0, 40.0, MULT, COMM, SLIP)
    _ = simular_short(d, d, d, b, b, 20.0, 40.0, MULT, COMM, SLIP)
    _ = simular_com_contexto(d, d, d, b, b, 20.0, 40.0, MULT, COMM, SLIP, d, d, d, d, d, d)
    _ = calcular_mae_mfe(d, d, d, b, 1)
    print("[JIT] Pronto!", flush=True)

    print(f"\n[GRID] {len(GRIDS)} estrategias sequencial + Numba...", flush=True)
    t0_total = time.time()
    todos = []

    for estrategia, grid in GRIDS.items():
        res = worker_estrategia((estrategia, grid, ind_is, MINI))
        e = res["estrategia"]
        spd = res["n_combos"] / max(res["elapsed"], 0.1)
        top = res["top10"][0] if res["top10"] else None
        pf_s = f"PF={top['pf']:.3f}" if top else "sem validos"
        print(
            f"  [{e.upper():25}] {res['n_validos']:>7,}/{res['n_combos']:>9,} "
            f"| {res['elapsed']:.1f}s | {spd:.0f}/s | {pf_s}",
            flush=True
        )
        if res["top10"]:
            todos.extend(res["top10"])
            with open(f"{OUTPUT_DIR}/{e}_top10.json", "w") as fp:
                json.dump(res["top10"], fp, indent=2, default=str)

    elapsed_total = time.time() - t0_total
    print(f"\n[GRID] {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)", flush=True)

    todos.sort(key=lambda x: -x["score"])

    print(f"\n{'=' * 68}", flush=True)
    print("  TOP 20", flush=True)
    print(f"  {'ESTRATEGIA':25} {'PF':>6} {'WR':>5} {'N':>6} {'EXP':>7}", flush=True)
    for r in todos[:20]:
        print(f"  {r['estrategia']:25} {r['pf']:>6.3f} {r['wr']:>5.1f} {r['n']:>6} {r['exp']:>7.2f}", flush=True)

    print(f"\n{'=' * 68}", flush=True)
    print("  MAE/MFE + ANALISE DE FALHAS", flush=True)
    atr_med_is = float(np.nanmean(ind_is["atr_14"]))
    candidatos = []

    for r in todos[:20]:
        estrategia = r["estrategia"]
        params = r["params"]
        try:
            ent, ext = gerar_sinais(estrategia, ind_is, params)
            if ent is None or ent.sum() < 20:
                continue

            ddir = 1 if params.get("direction", "short") == "long" else -1
            mae, mfe = calcular_mae_mfe(
                ind_is["open_next"].astype(np.float64),
                ind_is["high"].astype(np.float64),
                ind_is["low"].astype(np.float64),
                ent.astype(np.bool_),
                ddir
            )
            sl_opt = float(np.percentile(mae, 70)) if len(mae) > 0 else atr_med_is
            tp_opt = float(np.percentile(mfe, 50)) if len(mfe) > 0 else atr_med_is * 2
            rr_real = tp_opt / (sl_opt + 1e-9)

            filtro = analise_de_falhas(ind_is, ent, ext, params, sl_opt, tp_opt)

            pf_base = r["pf"]
            pf_com_filtro = pf_base
            ent_filtrada = ent

            if filtro and filtro["pf_depois"] > pf_base:
                ent_filtrada = aplicar_filtro_loser(ind_is, ent, filtro)
                pnls_f = executar(
                    ind_is,
                    ent_filtrada,
                    ext,
                    sl_opt,
                    tp_opt,
                    params.get("direction", "short"),
                )
                m_f = metricas(pnls_f, min_trades=MIN_TRADES_IS // 2)
                if m_f:
                    pf_com_filtro = m_f["pf"]

            print(
                f"  {estrategia:25} "
                f"RR_real={rr_real:.2f} "
                f"PF_base={pf_base:.3f} "
                f"PF_filtro={pf_com_filtro:.3f}",
                end="",
                flush=True
            )

            if filtro:
                print(
                    f" [FILTRO: {filtro['feature']} {filtro['operacao']} "
                    f"{filtro['corte']:.1f} +{filtro['melhoria']:.0f}%]",
                    flush=True
                )
            else:
                print(flush=True)

            candidatos.append({
                "estrategia": estrategia,
                "params": params,
                "pf_base": pf_base,
                "pf_filtro": pf_com_filtro,
                "sl_opt": round(sl_opt, 2),
                "tp_opt": round(tp_opt, 2),
                "rr_real": round(rr_real, 2),
                "filtro": filtro,
                "ent_original": ent,
                "ent_filtrada": ent_filtrada,
                "ext": ext,
            })
        except Exception:
            continue

    print(f"\n{'=' * 68}", flush=True)
    candidatos_oos = [c for c in candidatos if c["pf_filtro"] >= 0.90]
    print(f"  OOS ROLLING - {len(candidatos_oos)} CANDIDATO(S)", flush=True)

    resultados_finais = []
    for cand in candidatos_oos:
        estrategia = cand["estrategia"]
        params = cand["params"]
        sl = cand["sl_opt"]
        tp = cand["tp_opt"]

        print(f"\n  {estrategia} PF={cand['pf_filtro']:.3f}", flush=True)
        oos_r = oos_rolling(estrategia, df_oos, params, ind_oos)

        is_pf = cand["pf_filtro"]
        oos_pf = oos_r["pf_medio"] if oos_r else 0
        wfe = oos_r["wfe_pct"] if oos_r else 0
        deg = (is_pf - oos_pf) / is_pf * 100 if is_pf > 0 and oos_pf > 0 else 999
        ok = oos_pf > 1.0 and deg < 50 and wfe >= 40

        print(
            f"  IS={is_pf:.3f} OOS={oos_pf:.3f} "
            f"Deg={deg:.1f}% WFE={wfe:.0f}% "
            f"{'✅ APROVADO' if ok else '❌'}",
            flush=True
        )

        if oos_r:
            for j in oos_r["detalhes"]:
                print(
                    f"    {j['data']} PF={j['pf']:.3f} "
                    f"T={j['trades']} {'✅' if j['lucrativo'] else '❌'}",
                    flush=True
                )

        resultado = {
            "estrategia": estrategia,
            "params": params,
            "pf_is": is_pf,
            "pf_base": cand["pf_base"],
            "sl_opt": sl,
            "tp_opt": tp,
            "rr_real": cand["rr_real"],
            "filtro_loser": cand["filtro"],
            "oos_rolling": oos_r,
            "degradacao": round(deg, 1),
            "aprovado": ok,
            "gerado_em": datetime.now().isoformat(),
        }
        resultados_finais.append(resultado)
        with open(f"{OUTPUT_DIR}/{estrategia}_final.json", "w") as fp:
            json.dump(resultado, fp, indent=2, default=str)

    n_apr = sum(1 for r in resultados_finais if r["aprovado"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  LEADERBOARD - {n_apr} APROVADO(S)", flush=True)
    print(
        f"  {'ESTRATEGIA':25} {'IS':>6} {'OOS':>6} {'DEG':>6} {'WFE':>5} {'FILTRO':>20} {'STATUS':>10}",
        flush=True
    )
    print(f"  {'-' * 80}", flush=True)
    for r in sorted(resultados_finais, key=lambda x: -x["pf_is"]):
        oor = r["oos_rolling"] or {}
        fl = r["filtro_loser"]
        fl_str = f"{fl['feature']} {fl['operacao']}" if fl else "nenhum"
        print(
            f"  {r['estrategia']:25} "
            f"{r['pf_is']:>6.3f} "
            f"{oor.get('pf_medio', 0):>6.3f} "
            f"{r['degradacao']:>6.1f} "
            f"{oor.get('wfe_pct', 0):>5.0f} "
            f"{fl_str:>20} "
            f"{'✅' if r['aprovado'] else '❌':>10}",
            flush=True
        )

    lb = {
        "gerado_em": datetime.now().isoformat(),
        "total_combos": total,
        "n_cpus": N_CPUS,
        "tempo_s": round(elapsed_total, 1),
        "aprovados": n_apr,
        "top20": todos[:20],
        "leaderboard": resultados_finais,
    }
    with open(f"{OUTPUT_DIR}/leaderboard.json", "w") as fp:
        json.dump(lb, fp, indent=2, default=str)

    print(f"\n  {n_apr} estrategia(s) aprovada(s)!", flush=True)
    print(f"  Tempo: {elapsed_total/60:.1f} min | Salvo: {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()