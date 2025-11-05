import math
from typing import Dict, Any, List, Tuple
from datetime import datetime
import streamlit as st

# ============================================================
# FUNZIONI DI BASE
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def entropia_poisson(lam: float, max_k: int = 15) -> float:
    e = 0.0
    for k in range(max_k + 1):
        p = poisson_pmf(k, lam)
        if p > 0:
            e -= p * math.log2(p)
    return e

def decimali_a_prob(odds: float) -> float:
    return 1 / odds if odds and odds > 0 else 0.0

def normalize_1x2_from_odds(o1: float, ox: float, o2: float) -> Tuple[float, float, float]:
    p1 = 1 / o1 if o1 > 0 else 0.0
    px = 1 / ox if ox > 0 else 0.0
    p2 = 1 / o2 if o2 > 0 else 0.0
    tot = p1 + px + p2
    if tot == 0:
        return 0.33, 0.34, 0.33
    return p1 / tot, px / tot, p2 / tot

def gol_attesi_migliorati(spread: float, total: float,
                          p1: float, p2: float) -> Tuple[float, float]:
    if total < 2.25:
        total_eff = total * 1.03
    elif total > 3.0:
        total_eff = total * 0.97
    else:
        total_eff = total
    base = total_eff / 2.0
    diff = spread / 2.0
    fatt_int = 1 + (total_eff - 2.5) * 0.15
    lh = (base - diff) * fatt_int
    la = (base + diff) * fatt_int
    fatt_dir = ((p1 - p2) * 0.2) + 1.0
    lh *= fatt_dir
    la /= fatt_dir
    return max(lh, 0.05), max(la, 0.05)

def blend_lambda_market_xg(lambda_market_home: float,
                           lambda_market_away: float,
                           xg_for_home: float,
                           xg_against_home: float,
                           xg_for_away: float,
                           xg_against_away: float,
                           w_market: float = 0.6) -> Tuple[float, float]:
    xg_home_est = (xg_for_home + xg_against_away) / 2
    xg_away_est = (xg_for_away + xg_against_home) / 2
    lh = w_market * lambda_market_home + (1 - w_market) * xg_home_est
    la = w_market * lambda_market_away + (1 - w_market) * xg_away_est
    return max(lh, 0.05), max(la, 0.05)

def max_goals_adattivo(lh: float, la: float) -> int:
    return max(8, int((lh + la) * 2.5))

def tau_dixon_coles(h: int, a: int, lh: float, la: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1 - (lh * la * rho)
    elif h == 0 and a == 1:
        return 1 + (lh * rho)
    elif h == 1 and a == 0:
        return 1 + (la * rho)
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0

def build_score_matrix(lh: float, la: float, rho: float) -> List[List[float]]:
    mg = max_goals_adattivo(lh, la)
    mat: List[List[float]] = []
    for h in range(mg + 1):
        row = []
        for a in range(mg + 1):
            p = poisson_pmf(h, lh) * poisson_pmf(a, la)
            p *= tau_dixon_coles(h, a, lh, la, rho)
            row.append(p)
        mat.append(row)
    tot = sum(sum(r) for r in mat)
    mat = [[p / tot for p in r] for r in mat]
    return mat

def calc_match_result_from_matrix(mat: List[List[float]]) -> Tuple[float, float, float]:
    p_home = p_draw = p_away = 0.0
    mg = len(mat) - 1
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            if h > a:
                p_home += p
            elif h < a:
                p_away += p
            else:
                p_draw += p
    tot = p_home + p_draw + p_away
    return p_home / tot, p_draw / tot, p_away / tot

def calc_over_under_from_matrix(mat: List[List[float]], soglia: float) -> Tuple[float, float]:
    over = 0.0
    mg = len(mat) - 1
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h + a > soglia:
                over += mat[h][a]
    return over, 1 - over

def calc_bt_ts_from_matrix(mat: List[List[float]]) -> float:
    mg = len(mat) - 1
    return sum(mat[h][a] for h in range(1, mg + 1) for a in range(1, mg + 1))

def calc_gg_over25_from_matrix(mat: List[List[float]]) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            if h + a >= 3:
                s += mat[h][a]
    return s

def prob_pari_dispari_from_matrix(mat: List[List[float]]) -> Tuple[float, float]:
    mg = len(mat) - 1
    even = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            if (h + a) % 2 == 0:
                even += mat[h][a]
    return even, 1 - even

def prob_clean_sheet_from_matrix(mat: List[List[float]]) -> Tuple[float, float]:
    mg = len(mat) - 1
    cs_home = sum(mat[h][0] for h in range(mg + 1))
    cs_away = sum(mat[0][a] for a in range(mg + 1))
    return cs_away, cs_home

def dist_gol_da_matrice(mat: List[List[float]]):
    mg = len(mat) - 1
    dh = [0.0] * (mg + 1)
    da = [0.0] * (mg + 1)
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            dh[h] += p
            da[a] += p
    return dh, da

def prob_multigol_from_dist(dist: List[float], gmin: int, gmax: int) -> float:
    s = 0.0
    for k in range(gmin, gmax + 1):
        if k < len(dist):
            s += dist[k]
    return s

def combo_multigol_filtrata(multigol_casa: dict, multigol_away: dict, soglia: float = 0.5):
    out = []
    for kc, pc in multigol_casa.items():
        for ka, pa in multigol_away.items():
            p = pc * pa
            if p >= soglia:
                out.append({"combo": f"Casa {kc} + Ospite {ka}", "prob": p})
    out.sort(key=lambda x: x["prob"], reverse=True)
    return out

def prob_esito_over_from_matrix(mat: List[List[float]], esito: str, soglia: float) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h + a <= soglia:
                continue
            p = mat[h][a]
            if esito == '1' and h > a:
                s += p
            elif esito == 'X' and h == a:
                s += p
            elif esito == '2' and h < a:
                s += p
    return s

def prob_dc_over_from_matrix(mat: List[List[float]], dc: str, soglia: float) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h + a <= soglia:
                continue
            p = mat[h][a]
            if dc == '1X' and h >= a:
                s += p
            elif dc == 'X2' and a >= h:
                s += p
            elif dc == '12' and h != a:
                s += p
    return s

def prob_esito_btts_from_matrix(mat: List[List[float]], esito: str) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            if esito == '1' and h > a:
                s += p
            elif esito == 'X' and h == a:
                s += p
            elif esito == '2' and h < a:
                s += p
    return s

def prob_dc_btts_from_matrix(mat: List[List[float]], dc: str) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            ok = False
            if dc == '1X' and h >= a:
                ok = True
            elif dc == 'X2' and a >= h:
                ok = True
            elif dc == '12' and h != a:
                ok = True
            if ok:
                s += p
    return s

def combo_over_ht_ft(lh: float, la: float) -> Dict[str, float]:
    soglie = [0.5, 1.5, 2.5, 3.5]
    out = {}
    for ht in soglie:
        lam_ht = (lh + la) * 0.5
        p_under_ht = sum(poisson_pmf(k, lam_ht) for k in range(int(ht) + 1))
        p_over_ht = 1 - p_under_ht
        for ft in soglie:
            lam_ft = lh + la
            p_under_ft = sum(poisson_pmf(k, lam_ft) for k in range(int(ft) + 1))
            p_over_ft = 1 - p_under_ft
            out[f"Over HT {ht} + Over FT {ft}"] = min(1.0, p_over_ht * p_over_ft)
    return out

def top_results_from_matrix(mat, top_n=10, soglia_min=0.005):
    mg = len(mat) - 1
    risultati = []
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            if p >= soglia_min:
                risultati.append((h, a, p * 100))
    risultati.sort(key=lambda x: x[2], reverse=True)
    return risultati[:top_n]

# ============================================================
# MODELLO COMPLETO
# ============================================================

def risultato_completo(spread: float, total: float,
                       odds_1: float, odds_x: float, odds_2: float,
                       odds_btts: float,
                       xg_for_home: float = None,
                       xg_against_home: float = None,
                       xg_for_away: float = None,
                       xg_against_away: float = None) -> Dict[str, Any]:

    p1, px, p2 = normalize_1x2_from_odds(odds_1, odds_x, odds_2)
    lh, la = gol_attesi_migliorati(spread, total, p1, p2)

    if (xg_for_home is not None and xg_against_home is not None and
        xg_for_away is not None and xg_against_away is not None):
        lh, la = blend_lambda_market_xg(
            lh, la,
            xg_for_home, xg_against_home,
            xg_for_away, xg_against_away,
            w_market=0.6
        )

    if odds_btts and odds_btts > 1:
        p_btts_market = 1 / odds_btts
        rho = 0.15 + (p_btts_market - 0.55) * 0.8
        rho = max(0.05, min(0.45, rho))
    else:
        rho = 0.15 + (px * 0.4)
        rho = max(0.05, min(0.4, rho))

    mat_ft = build_score_matrix(lh, la, rho)
    ratio_ht = 0.46 + 0.02 * (total - 2.5)
    mat_ht = build_score_matrix(lh * ratio_ht, la * ratio_ht, rho)

    p_home, p_draw, p_away = calc_match_result_from_matrix(mat_ft)

    over_15, under_15 = calc_over_under_from_matrix(mat_ft, 1.5)
    over_25, under_25 = calc_over_under_from_matrix(mat_ft, 2.5)
    over_35, under_35 = calc_over_under_from_matrix(mat_ft, 3.5)
    over_05_ht = 1 - mat_ht[0][0]

    btts = calc_bt_ts_from_matrix(mat_ft)
    gg_over25 = calc_gg_over25_from_matrix(mat_ft)

    even_ft, odd_ft = prob_pari_dispari_from_matrix(mat_ft)
    even_ht, odd_ht = prob_pari_dispari_from_matrix(mat_ht)

    cs_away, cs_home = prob_clean_sheet_from_matrix(mat_ft)
    clean_sheet_qualcuno = 1 - btts

    dist_home_ft, dist_away_ft = dist_gol_da_matrice(mat_ft)
    dist_home_ht, dist_away_ht = dist_gol_da_matrice(mat_ht)

    ranges = [(0,1),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,5)]
    multigol_home = {f"{a}-{b}": prob_multigol_from_dist(dist_home_ft, a, b) for a,b in ranges}
    multigol_away = {f"{a}-{b}": prob_multigol_from_dist(dist_away_ft, a, b) for a,b in ranges}
    multigol_home_ht = {f"{a}-{b}": prob_multigol_from_dist(dist_home_ht, a, b) for a,b in ranges}
    multigol_away_ht = {f"{a}-{b}": prob_multigol_from_dist(dist_away_ht, a, b) for a,b in ranges}

    combo_ft_filtrate = combo_multigol_filtrata(multigol_home, multigol_away, 0.5)
    combo_ht_filtrate = combo_multigol_filtrata(multigol_home_ht, multigol_away_ht, 0.5)

    dc = {
        "DC Casa o Pareggio": p_home + p_draw,
        "DC Trasferta o Pareggio": p_away + p_draw,
        "DC Casa o Trasferta": p_home + p_away
    }

    mg = len(mat_ft) - 1
    marg2 = marg3 = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat_ft[h][a]
            if h - a >= 2:
                marg2 += p
            if h - a >= 3:
                marg3 += p

    combo_book = {
        "1 & Over 1.5": prob_esito_over_from_matrix(mat_ft, '1', 1.5),
        "1 & Over 2.5": prob_esito_over_from_matrix(mat_ft, '1', 2.5),
        "2 & Over 1.5": prob_esito_over_from_matrix(mat_ft, '2', 1.5),
        "2 & Over 2.5": prob_esito_over_from_matrix(mat_ft, '2', 2.5),
        "1X & Over 1.5": prob_dc_over_from_matrix(mat_ft, '1X', 1.5),
        "X2 & Over 1.5": prob_dc_over_from_matrix(mat_ft, 'X2', 1.5),
        "1X & Over 2.5": prob_dc_over_from_matrix(mat_ft, '1X', 2.5),
        "X2 & Over 2.5": prob_dc_over_from_matrix(mat_ft, 'X2', 2.5),
        "1X & BTTS": prob_dc_btts_from_matrix(mat_ft, '1X'),
        "X2 & BTTS": prob_dc_btts_from_matrix(mat_ft, 'X2'),
        "1 & BTTS": prob_esito_btts_from_matrix(mat_ft, '1'),
        "2 & BTTS": prob_esito_btts_from_matrix(mat_ft, '2'),
    }

    combo_ht_ft = combo_over_ht_ft(lh, la)

    top10 = top_results_from_matrix(mat_ft, 10, 0.005)

    ent_home = entropia_poisson(lh)
    ent_away = entropia_poisson(la)

    odds_prob = {
        "1": decimali_a_prob(odds_1),
        "X": decimali_a_prob(odds_x),
        "2": decimali_a_prob(odds_2)
    }
    scost = {
        "1": (p_home - odds_prob["1"]) * 100,
        "X": (p_draw - odds_prob["X"]) * 100,
        "2": (p_away - odds_prob["2"]) * 100
    }

    return {
        "lambda_home": lh,
        "lambda_away": la,
        "rho": rho,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "over_15": over_15,
        "under_15": under_15,
        "over_25": over_25,
        "under_25": under_25,
        "over_35": over_35,
        "under_35": under_35,
        "over_05_ht": over_05_ht,
        "btts": btts,
        "gg_over25": gg_over25,
        "even_ft": even_ft,
        "odd_ft": odd_ft,
        "even_ht": even_ht,
        "odd_ht": odd_ht,
        "cs_home": cs_home,
        "cs_away": cs_away,
        "clean_sheet_qualcuno": clean_sheet_qualcuno,
        "multigol_home": multigol_home,
        "multigol_away": multigol_away,
        "multigol_home_ht": multigol_home_ht,
        "multigol_away_ht": multigol_away_ht,
        "dc": dc,
        "marg2": marg2,
        "marg3": marg3,
        "combo_ft_filtrate": combo_ft_filtrate,
        "combo_ht_filtrate": combo_ht_filtrate,
        "combo_book": combo_book,
        "combo_ht_ft": combo_ht_ft,
        "top10": top10,
        "ent_home": ent_home,
        "ent_away": ent_away,
        "odds_prob": odds_prob,
        "scost": scost
    }

# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Modello Scommesse V5", layout="wide")

st.title("📊 Modello Scommesse V5 – Prematch completo")
st.caption(f"Aggiornato: {datetime.now().isoformat(timespec='seconds')}")

st.subheader("1. Linee di mercato")

col1, col2 = st.columns(2)
with col1:
    spread_ap = st.number_input("Spread apertura", value=-0.25, step=0.25)
    total_ap = st.number_input("Total apertura", value=2.5, step=0.25)
with col2:
    spread_co = st.number_input("Spread corrente", value=-0.5, step=0.25)
    total_co = st.number_input("Total corrente", value=2.25, step=0.25)

st.subheader("2. Quote principali")
colq1, colq2, colq3, colq4 = st.columns(4)
with colq1:
    odds_1 = st.number_input("Quota 1", value=1.80, step=0.01)
with colq2:
    odds_x = st.number_input("Quota X", value=3.60, step=0.01)
with colq3:
    odds_2 = st.number_input("Quota 2", value=4.50, step=0.01)
with colq4:
    odds_btts = st.number_input("Quota GG (BTTS sì)", value=1.95, step=0.01)

st.subheader("3. xG avanzati (facoltativo)")
colxh, colxa = st.columns(2)
with colxh:
    xg_tot_home = st.text_input("xG totali CASA", "")
    xga_tot_home = st.text_input("xGA totali CASA", "")
    record_home = st.text_input("Record/partite CASA (es. 5-3-2 o 10)", "")
with colxa:
    xg_tot_away = st.text_input("xG totali OSPITE", "")
    xga_tot_away = st.text_input("xGA totali OSPITE", "")
    record_away = st.text_input("Record/partite OSPITE", "")

def parse_xg_block(xg_tot_s: str, xga_tot_s: str, record_s: str):
    if xg_tot_s.strip() == "" or xga_tot_s.strip() == "" or record_s.strip() == "":
        return None, None
    try:
        xg_tot = float(xg_tot_s.replace(",", "."))
        xga_tot = float(xga_tot_s.replace(",", "."))
        if "-" in record_s:
            parts = record_s.split("-")
            matches = sum(int(p) for p in parts if p.strip() != "")
        else:
            matches = int(record_s.strip())
        if matches <= 0:
            return None, None
        return xg_tot / matches, xga_tot / matches
    except Exception:
        return None, None

xg_home_for, xg_home_against = parse_xg_block(xg_tot_home, xga_tot_home, record_home)
xg_away_for, xg_away_against = parse_xg_block(xg_tot_away, xga_tot_away, record_away)

if (xg_home_for is None or xg_home_against is None or
    xg_away_for is None or xg_away_against is None):
    st.warning("Modalità: BASE (solo spread/total/quote). Inserisci anche xG/xGA per la modalità avanzata.")
else:
    st.success("Modalità: AVANZATA (spread/total + quote + xG/xGA).")

if st.button("Calcola modello"):
    ris_ap = risultato_completo(
        spread_ap, total_ap,
        odds_1, odds_x, odds_2,
        0.0,
        xg_home_for, xg_home_against,
        xg_away_for, xg_away_against
    )
    ris_co = risultato_completo(
        spread_co, total_co,
        odds_1, odds_x, odds_2,
        odds_btts,
        xg_home_for, xg_home_against,
        xg_away_for, xg_away_against
    )

    st.success("Calcolo completato ✅")

    # 1. Probabilità principali
    with st.expander("1️⃣ Probabilità principali", expanded=True):
        st.write(f"BTTS: {ris_co['btts']*100:.1f}%")
        st.write(f"No Goal: {(1-ris_co['btts'])*100:.1f}%")
        st.write(f"GG + Over 2.5: {ris_co['gg_over25']*100:.1f}%")

    # 2. Esito finale e parziale
    with st.expander("2️⃣ Esito finale e parziale"):
        st.write(f"Vittoria Casa: {ris_co['p_home']*100:.1f}%")
        st.write(f"Pareggio: {ris_co['p_draw']*100:.1f}%")
        st.write(f"Vittoria Trasferta: {ris_co['p_away']*100:.1f}%")
        st.write("Double Chance:")
        for k, v in ris_co["dc"].items():
            st.write(f"- {k}: {v*100:.1f}%")
        st.write("Confronto apertura → corrente:")
        st.write(f"1: {ris_co['p_home']*100:.1f}% (apertura {ris_ap['p_home']*100:.1f}%)")
        st.write(f"X: {ris_co['p_draw']*100:.1f}% (apertura {ris_ap['p_draw']*100:.1f}%)")
        st.write(f"2: {ris_co['p_away']*100:.1f}% (apertura {ris_ap['p_away']*100:.1f}%)")

    # 3. Over / Under
    with st.expander("3️⃣ Over / Under"):
        st.write(f"Over 1.5: {ris_co['over_15']*100:.1f}%")
        st.write(f"Under 1.5: {ris_co['under_15']*100:.1f}%")
        st.write(f"Over 2.5: {ris_co['over_25']*100:.1f}%")
        st.write(f"Under 2.5: {ris_co['under_25']*100:.1f}%")
        st.write(f"Over 3.5: {ris_co['over_35']*100:.1f}%")
        st.write(f"Under 3.5: {ris_co['under_35']*100:.1f}%")
        st.write(f"Over 0.5 HT: {ris_co['over_05_ht']*100:.1f}%")

    # 4. Gol pari / dispari
    with st.expander("4️⃣ Gol pari/dispari"):
        st.write(f"Gol pari FT: {ris_co['even_ft']*100:.1f}%")
        st.write(f"Gol dispari FT: {ris_co['odd_ft']*100:.1f}%")
        st.write(f"Gol pari HT: {ris_co['even_ht']*100:.1f}%")
        st.write(f"Gol dispari HT: {ris_co['odd_ht']*100:.1f}%")

    # 5. Clean Sheet e Gol Attesi
    with st.expander("5️⃣ Clean Sheet, gol attesi e info modello"):
        st.write(f"Clean Sheet Casa: {ris_co['cs_home']*100:.1f}%")
        st.write(f"Clean Sheet Trasferta: {ris_co['cs_away']*100:.1f}%")
        st.write(f"Clean Sheet qualcuno (No Goal): {ris_co['clean_sheet_qualcuno']*100:.1f}%")
        st.write(f"Gol attesi Casa (λ): {ris_co['lambda_home']:.3f}")
        st.write(f"Gol attesi Trasferta (λ): {ris_co['lambda_away']:.3f}")
        st.write(f"Entropia Casa: {ris_co['ent_home']:.3f}")
        st.write(f"Entropia Trasferta: {ris_co['ent_away']:.3f}")

    # 6. Multigol Casa
    with st.expander("6️⃣ Multigol Casa"):
        st.write({k: f"{v*100:.1f}%" for k, v in ris_co["multigol_home"].items()})

    # 7. Multigol Trasferta
    with st.expander("7️⃣ Multigol Trasferta"):
        st.write({k: f"{v*100:.1f}%" for k, v in ris_co["multigol_away"].items()})

    # 8. Vittoria con margine
    with st.expander("8️⃣ Vittoria con margine"):
        st.write(f"Vittoria casa almeno 2 gol scarto: {ris_co['marg2']*100:.1f}%")
        st.write(f"Vittoria casa almeno 3 gol scarto: {ris_co['marg3']*100:.1f}%")

    # 9. Combo mercati
    with st.expander("9️⃣ Combo mercati (1&Over, DC+GG, 1&BTTS, ecc.)"):
        for k, v in ris_co["combo_book"].items():
            st.write(f"{k}: {v*100:.1f}%")

    # 10. Risultati esatti
    with st.expander("🔟 Top 10 risultati esatti"):
        for h, a, p in ris_co["top10"]:
            st.write(f"{h}-{a}: {p:.1f}%")

    # 11. Combo Multigol filtrate
    with st.expander("1️⃣1️⃣ Combo Multigol Filtrate (>=50%)"):
        for c in ris_co["combo_ft_filtrate"]:
            st.write(f"{c['combo']}: {c['prob']*100:.1f}%")

    # 12. Combo Over HT + Over FT
    with st.expander("1️⃣2️⃣ Combo Over HT + Over FT"):
        for k, v in ris_co["combo_ht_ft"].items():
            st.write(f"{k}: {v*100:.1f}%")

    # 13. Statistiche avanzate
    with st.expander("1️⃣3️⃣ Statistiche avanzate e scostamenti vs quote"):
        st.write("Probabilità implicite dalle quote:")
        st.write({k: f"{v*100:.2f}%" for k, v in ris_co["odds_prob"].items()})
        st.write("Scostamento modello vs quote (in punti percentuali):")
        st.write({k: f"{v:+.2f} pp" for k, v in ris_co["scost"].items()})
else:
    st.info("Compila i campi e premi **Calcola modello**.")
