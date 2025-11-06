import math
from typing import Dict, Any, List, Tuple
from datetime import datetime, date, timezone, timedelta
import pandas as pd
import os
import requests
import streamlit as st

# ============================================================
#                CONFIGURAZIONE BASE
# ============================================================
# tua key di The Odds API
API_KEY_ODDS = "dd98fa8bb347a59f78867a4fe680e1e8"
ARCHIVE_FILE = "storico_analisi.csv"

# elenco bookmaker che proviamo a mediare
PREFERRED_BOOKMAKERS = [
    "pinnacle",
    "bet365",
    "williamhill",
    "unibet",
    "ladbrokes",
    "betfair",
    "marathonbet",
]

# mappa nomi campionati -> sport_key di The Odds API
SPORT_KEYS = {
    "Serie A": "soccer_italy_serie_a",
    "Premier League": "soccer_epl",
    "Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Champions League": "soccer_uefa_champs_league",
}

# ============================================================
#                FUNZIONI MODELLO
# (qui non tocchiamo: è il motore che già funzionava)
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
    p1 = 1 / o1 if o1 and o1 > 0 else 0.0
    px = 1 / ox if ox and ox > 0 else 0.0
    p2 = 1 / o2 if o2 and o2 > 0 else 0.0
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
    cs_away = sum(mat[0][a] for a in range(mg + 1))
    cs_home = sum(mat[h][0] for h in range(mg + 1))
    return cs_home, cs_away

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

    # rho guidato da quota GG se c'è
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

    cs_home, cs_away = prob_clean_sheet_from_matrix(mat_ft)
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
#        CHECK DI AFFIDABILITÀ
# ============================================================

def controlli_affidabilita_base(
    ris_co: dict,
    spread_ap: float,
    spread_co: float,
    total_ap: float,
    total_co: float,
    has_xg: bool,
    odds_btts: float,
    odds_over25: float,
    odds_under25: float,
    league_name: str = "",
    fixture_datetime: datetime = None,
) -> int:
    penalty = 0
    if ris_co["btts"] > 0.70 and (ris_co["p_home"] > 0.65 or ris_co["p_away"] > 0.65):
        penalty += 7
    if total_co < 2.25 and ris_co["over_25"] > 0.62:
        penalty += 5
    if not has_xg:
        penalty += 5
    if not odds_btts or odds_btts <= 1:
        penalty += 4
    if (not odds_over25 or odds_over25 <= 1) and (not odds_under25 or odds_under25 <= 1):
        penalty += 3
    if abs(spread_ap - spread_co) > 0.25:
        penalty += 5
    if abs(total_ap - total_co) > 0.25:
        penalty += 4
    return penalty

def compute_confidence(
    ris_co: dict,
    spread_ap: float,
    spread_co: float,
    total_ap: float,
    total_co: float,
    has_xg: bool,
    odds_btts: float
) -> int:
    score = 100
    ent_med = (ris_co["ent_home"] + ris_co["ent_away"]) / 2
    if ent_med > 2.2:
        score -= 15
    elif ent_med > 2.0:
        score -= 5
    if abs(spread_ap - spread_co) > 0.25:
        score -= 10
    if abs(total_ap - total_co) > 0.25:
        score -= 8
    if not has_xg:
        score -= 8
    if not odds_btts or odds_btts <= 1:
        score -= 4
    if ris_co["btts"] > 0.7 and (ris_co["p_home"] > 0.65 or ris_co["p_away"] > 0.65):
        score -= 8
    return max(0, min(100, score))

# ============================================================
#     FUNZIONI API – THE ODDS API (con confronto 5-7 book)
# ============================================================

def oddsapi_get_events(sport_key: str) -> list:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY_ODDS,
        "regions": "eu,uk",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        return data
    except Exception:
        return []

def oddsapi_extract_avg_odds(event: dict,
                             wanted_books: List[str]) -> Dict[str, float]:
    """
    Prende un singolo evento The Odds API e prova a ricostruire:
    - quota media 1
    - quota media X
    - quota media 2
    - quota media Over 2.5 / Under 2.5 (se c'è)
    usando solo i bookmaker in wanted_books se presenti.
    """
    out = {
        "odds_1": None,
        "odds_x": None,
        "odds_2": None,
        "odds_over25": None,
        "odds_under25": None,
    }

    bks = event.get("bookmakers", [])
    if not bks:
        return out

    home_team = event.get("home_team", "").lower()
    away_team = event.get("away_team", "").lower()

    list_1 = []
    list_x = []
    list_2 = []
    list_over = []
    list_under = []

    for bk in bks:
        bk_key = bk.get("key", "").lower()
        if bk_key not in wanted_books:
            continue

        for m in bk.get("markets", []):
            mkey = m.get("key", "")
            # 1X2
            if mkey == "h2h":
                for o in m.get("outcomes", []):
                    name = o.get("name", "").lower()
                    price = float(o.get("price", 0))
                    if price < 1.05 or price > 15:
                        continue
                    if name == home_team:
                        list_1.append(price)
                    elif name == away_team:
                        list_2.append(price)
                    elif name in ["draw", "pareggio"]:
                        list_x.append(price)

            # totals
            elif mkey == "totals":
                point = m.get("point", None)
                # se è 2.5 ci interessa
                if point is not None and abs(float(point) - 2.5) < 0.001:
                    for o in m.get("outcomes", []):
                        desc = o.get("name", "").lower()
                        price = float(o.get("price", 0))
                        if price < 1.05 or price > 15:
                            continue
                        if "over" in desc:
                            list_over.append(price)
                        elif "under" in desc:
                            list_under.append(price)

    def avg_or_none(arr):
        return round(sum(arr)/len(arr), 2) if arr else None

    out["odds_1"] = avg_or_none(list_1)
    out["odds_x"] = avg_or_none(list_x)
    out["odds_2"] = avg_or_none(list_2)
    out["odds_over25"] = avg_or_none(list_over)
    out["odds_under25"] = avg_or_none(list_under)

    return out

# ============================================================
#                   STREAMLIT APP
# ============================================================

# stato
if "events" not in st.session_state:
    st.session_state.events = []
if "selected_event_key" not in st.session_state:
    st.session_state.selected_event_key = None
if "odds_from_api" not in st.session_state:
    st.session_state.odds_from_api = {}

st.subheader("0. Carica partite da The Odds API")
col_s0, col_s1 = st.columns([2,1])
with col_s0:
    league_label = st.selectbox("Campionato", list(SPORT_KEYS.keys()), index=0)
with col_s1:
    if st.button("📡 Carica partite"):
        sport_key = SPORT_KEYS[league_label]
        evs = oddsapi_get_events(sport_key)
        st.session_state.events = evs
        st.session_state.selected_event_key = None
        if not evs:
            st.warning("Nessuna partita trovata oppure limite API.")
        else:
            st.success(f"Trovate {len(evs)} partite")

events = st.session_state.events

event_options = []
event_map = {}
if events:
    for idx, ev in enumerate(events):
        key = f"{ev['home_team']} vs {ev['away_team']}"
        event_options.append(key)
        event_map[key] = ev

if event_options:
    default_index = 0
    if st.session_state.selected_event_key in event_options:
        default_index = event_options.index(st.session_state.selected_event_key)
    selected_key = st.selectbox("Partita trovata:", event_options, index=default_index)
    st.session_state.selected_event_key = selected_key

    ev = event_map[selected_key]
    parsed_odds = oddsapi_extract_avg_odds(ev, PREFERRED_BOOKMAKERS)
    st.session_state.odds_from_api = parsed_odds

    st.info(f"Partita selezionata: **{selected_key}** (media su {len(PREFERRED_BOOKMAKERS)} book)")
    st.write("Quote trovate (solo se il book le aveva):")
    st.json(parsed_odds)
else:
    parsed_odds = {}

st.markdown("---")

# ============================================================
# 1. INPUT MANUALE (CUORE DEL MODELLO)
# ============================================================

st.subheader("1. Dati partita (manuali + auto)")

match_name = st.text_input("Nome partita", value=st.session_state.selected_event_key or "")

col_ap1, col_ap2 = st.columns(2)
with col_ap1:
    spread_ap = st.number_input("Spread apertura", value=0.0, step=0.25)
with col_ap2:
    total_ap = st.number_input("Total apertura", value=2.5, step=0.25)

col_co1, col_co2, col_co3 = st.columns(3)
with col_co1:
    spread_co = st.number_input("Spread corrente", value=0.0, step=0.25)
    odds_1 = st.number_input(
        "Quota 1",
        value=parsed_odds.get("odds_1") or st.session_state.odds_from_api.get("odds_1") or 1.80,
        step=0.01
    )
with col_co2:
    total_co = st.number_input("Total corrente", value=2.5, step=0.25)
    odds_x = st.number_input(
        "Quota X",
        value=parsed_odds.get("odds_x") or st.session_state.odds_from_api.get("odds_x") or 3.50,
        step=0.01
    )
with col_co3:
    odds_2 = st.number_input(
        "Quota 2",
        value=parsed_odds.get("odds_2") or st.session_state.odds_from_api.get("odds_2") or 4.50,
        step=0.01
    )
    odds_btts = st.number_input("Quota GG (BTTS sì)", value=1.95, step=0.01)

st.subheader("2. Quote Over/Under (opzionali)")
col_ou1, col_ou2 = st.columns(2)
with col_ou1:
    odds_over25 = st.number_input(
        "Quota Over 2.5 (opzionale)",
        value=parsed_odds.get("odds_over25") or 0.0,
        step=0.01
    )
with col_ou2:
    odds_under25 = st.number_input(
        "Quota Under 2.5 (opzionale)",
        value=parsed_odds.get("odds_under25") or 0.0,
        step=0.01
    )

st.subheader("3. xG avanzati (opzionali)")
col_xg1, col_xg2 = st.columns(2)
with col_xg1:
    xg_tot_home = st.text_input("xG totali CASA", "")
    xga_tot_home = st.text_input("xGA totali CASA", "")
    partite_home = st.text_input("Partite giocate CASA (es. 10 o 5-3-2)", "")
with col_xg2:
    xg_tot_away = st.text_input("xG totali OSPITE", "")
    xga_tot_away = st.text_input("xGA totali OSPITE", "")
    partite_away = st.text_input("Partite giocate OSPITE", "")

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

xg_home_for, xg_home_against = parse_xg_block(xg_tot_home, xga_tot_home, partite_home)
xg_away_for, xg_away_against = parse_xg_block(xg_tot_away, xga_tot_away, partite_away)

has_xg = not (
    xg_home_for is None or xg_home_against is None or
    xg_away_for is None or xg_away_against is None
)

if not has_xg:
    st.info("Modalità: BASE (solo spread/total/quote).")
else:
    st.success("Modalità: AVANZATA (spread/total + quote + xG/xGA).")

# ============================================================
#                    CALCOLO MODELLO
# ============================================================

if st.button("CALCOLA MODELLO"):
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

    aff = 100
    if abs(spread_ap - spread_co) > 0.25:
        aff -= 15
    if abs(total_ap - total_co) > 0.25:
        aff -= 10
    ent_media = (ris_co["ent_home"] + ris_co["ent_away"]) / 2
    if ent_media > 2.2:
        aff -= 15
    if not has_xg:
        aff -= 10

    smart_pen = controlli_affidabilita_base(
        ris_co,
        spread_ap, spread_co,
        total_ap, total_co,
        has_xg,
        odds_btts,
        odds_over25,
        odds_under25,
        "",
        None
    )
    aff -= smart_pen
    aff = max(0, min(100, aff))

    confidence = compute_confidence(
        ris_co,
        spread_ap, spread_co,
        total_ap, total_co,
        has_xg,
        odds_btts
    )

    st.success("Calcolo completato ✅")
    st.subheader("⭐ Sintesi Match")
    st.write(f"Affidabilità del match: **{aff}/100**")
    st.write(f"Confidence Engine: **{confidence}/100**")

    # VALUE FINDER
    st.subheader("💰 Value Finder + EV")
    soglia_pp = 5.0
    rows = []
    value_markets = []

    for lab, p_mod, p_book, odd in [
        ("1", ris_co["p_home"], ris_co["odds_prob"]["1"], odds_1),
        ("X", ris_co["p_draw"], ris_co["odds_prob"]["X"], odds_x),
        ("2", ris_co["p_away"], ris_co["odds_prob"]["2"], odds_2),
    ]:
        diff = (p_mod - p_book) * 100
        ev = p_mod * odd - 1 if odd and odd > 0 else None
        is_val = diff >= soglia_pp
        if is_val:
            value_markets.append(f"1X2 {lab}")
        rows.append({
            "Mercato": "1X2",
            "Esito": lab,
            "Prob modello %": round(p_mod*100, 2),
            "Prob quota %": round(p_book*100, 2),
            "Δ pp": round(diff, 2),
            "EV %": round(ev*100, 2) if ev is not None else None,
            "Value?": "✅" if is_val else ""
        })

    prob_gg_model = ris_co["btts"]
    prob_gg_book = decimali_a_prob(odds_btts)
    if prob_gg_book > 0:
        diff_gg = (prob_gg_model - prob_gg_book) * 100
        ev_gg = prob_gg_model * odds_btts - 1
        is_val = diff_gg >= soglia_pp
        if is_val:
            value_markets.append("BTTS Sì")
        rows.append({
            "Mercato": "GG/NG",
            "Esito": "GG",
            "Prob modello %": round(prob_gg_model*100, 2),
            "Prob quota %": round(prob_gg_book*100, 2),
            "Δ pp": round(diff_gg, 2),
            "EV %": round(ev_gg*100, 2),
            "Value?": "✅" if is_val else ""
        })
    else:
        rows.append({
            "Mercato": "GG/NG",
            "Esito": "GG",
            "Prob modello %": round(prob_gg_model*100, 2),
            "Prob quota %": None,
            "Δ pp": None,
            "EV %": None,
            "Value?": "Quota GG non inserita"
        })

    prob_over_model = ris_co["over_25"]
    prob_under_model = ris_co["under_25"]

    if odds_over25 and odds_over25 > 1:
        prob_over_book = decimali_a_prob(odds_over25)
        diff_over = (prob_over_model - prob_over_book) * 100
        ev_over = prob_over_model * odds_over25 - 1
        is_val = diff_over >= soglia_pp
        if is_val:
            value_markets.append("Over 2.5")
        rows.append({
            "Mercato": "Over/Under 2.5",
            "Esito": "Over 2.5",
            "Prob modello %": round(prob_over_model*100, 2),
            "Prob quota %": round(prob_over_book*100, 2),
            "Δ pp": round(diff_over, 2),
            "EV %": round(ev_over*100, 2),
            "Value?": "✅" if is_val else ""
        })
    else:
        rows.append({
            "Mercato": "Over/Under 2.5",
            "Esito": "Over 2.5",
            "Prob modello %": round(prob_over_model*100, 2),
            "Prob quota %": None,
            "Δ pp": None,
            "EV %": None,
            "Value?": "Quota non inserita"
        })

    if odds_under25 and odds_under25 > 1:
        prob_under_book = decimali_a_prob(odds_under25)
        diff_under = (prob_under_model - prob_under_book) * 100
        ev_under = prob_under_model * odds_under25 - 1
        is_val = diff_under >= soglia_pp
        if is_val:
            value_markets.append("Under 2.5")
        rows.append({
            "Mercato": "Over/Under 2.5",
            "Esito": "Under 2.5",
            "Prob modello %": round(prob_under_model*100, 2),
            "Prob quota %": round(prob_under_book*100, 2),
            "Δ pp": round(diff_under, 2),
            "EV %": round(ev_under*100, 2),
            "Value?": "✅" if is_val else ""
        })
    else:
        rows.append({
            "Mercato": "Over/Under 2.5",
            "Esito": "Under 2.5",
            "Prob modello %": round(prob_under_model*100, 2),
            "Prob quota %": None,
            "Δ pp": None,
            "EV %": None,
            "Value?": "Quota non inserita"
        })

    df_value = pd.DataFrame(rows)
    st.dataframe(df_value)

    # PAPIRO COMPLETO
    with st.expander("1️⃣ Probabilità principali"):
        st.write(f"BTTS: {ris_co['btts']*100:.1f}%")
        st.write(f"No Goal: {(1-ris_co['btts'])*100:.1f}%")
        st.write(f"GG + Over 2.5: {ris_co['gg_over25']*100:.1f}%")

    with st.expander("2️⃣ Esito finale e parziale"):
        st.write(f"Vittoria Casa: {ris_co['p_home']*100:.1f}% (apertura {ris_ap['p_home']*100:.1f}%)")
        st.write(f"Pareggio: {ris_co['p_draw']*100:.1f}% (apertura {ris_ap['p_draw']*100:.1f}%)")
        st.write(f"Vittoria Trasferta: {ris_co['p_away']*100:.1f}% (apertura {ris_ap['p_away']*100:.1f}%)")
        st.write("Double Chance:")
        for k, v in ris_co["dc"].items():
            st.write(f"- {k}: {v*100:.1f}%")

    with st.expander("3️⃣ Over / Under"):
        st.write(f"Over 1.5: {ris_co['over_15']*100:.1f}%")
        st.write(f"Under 1.5: {ris_co['under_15']*100:.1f}%")
        st.write(f"Over 2.5: {ris_co['over_25']*100:.1f}%")
        st.write(f"Under 2.5: {ris_co['under_25']*100:.1f}%")
        st.write(f"Over 3.5: {ris_co['over_35']*100:.1f}%")
        st.write(f"Under 3.5: {ris_co['under_35']*100:.1f}%")
        st.write(f"Over 0.5 HT: {ris_co['over_05_ht']*100:.1f}%")

    with st.expander("4️⃣ Gol pari/dispari"):
        st.write(f"Gol pari FT: {ris_co['even_ft']*100:.1f}%")
        st.write(f"Gol dispari FT: {ris_co['odd_ft']*100:.1f}%")
        st.write(f"Gol pari HT: {ris_co['even_ht']*100:.1f}%")
        st.write(f"Gol dispari HT: {ris_co['odd_ht']*100:.1f}%")

    with st.expander("5️⃣ Clean sheet e info modello"):
        st.write(f"Clean Sheet Casa: {ris_co['cs_home']*100:.1f}%")
        st.write(f"Clean Sheet Trasferta: {ris_co['cs_away']*100:.1f}%")
        st.write(f"Clean Sheet qualcuno (No Goal): {ris_co['clean_sheet_qualcuno']*100:.1f}%")
        st.write(f"λ Casa: {ris_co['lambda_home']:.3f}")
        st.write(f"λ Trasferta: {ris_co['lambda_away']:.3f}")
        st.write(f"Entropia Casa: {ris_co['ent_home']:.3f}")
        st.write(f"Entropia Trasferta: {ris_co['ent_away']:.3f}")

    with st.expander("6️⃣ Multigol Casa"):
        st.write({k: f"{v*100:.1f}%" for k, v in ris_co["multigol_home"].items()})

    with st.expander("7️⃣ Multigol Trasferta"):
        st.write({k: f"{v*100:.1f}%" for k, v in ris_co["multigol_away"].items()})

    with st.expander("8️⃣ Vittoria con margine"):
        st.write(f"Vittoria casa almeno 2 gol scarto: {ris_co['marg2']*100:.1f}%")
        st.write(f"Vittoria casa almeno 3 gol scarto: {ris_co['marg3']*100:.1f}%")

    with st.expander("9️⃣ Combo mercati"):
        for k, v in ris_co["combo_book"].items():
            st.write(f"{k}: {v*100:.1f}%")

    with st.expander("🔟 Top 10 risultati esatti"):
        for h, a, p in ris_co["top10"]:
            st.write(f"{h}-{a}: {p:.1f}%")

    with st.expander("1️⃣1️⃣ Combo Multigol Filtrate (>=50%)"):
        for c in ris_co["combo_ft_filtrate"]:
            st.write(f"{c['combo']}: {c['prob']*100:.1f}%")

    with st.expander("1️⃣2️⃣ Combo Over HT + Over FT"):
        for k, v in ris_co["combo_ht_ft"].items():
            st.write(f"{k}: {v*100:.1f}%")

    # salvataggio storico
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "match": match_name,
        "match_date": date.today().isoformat(),
        "spread_ap": spread_ap,
        "total_ap": total_ap,
        "spread_co": spread_co,
        "total_co": total_co,
        "odds_1": odds_1,
        "odds_x": odds_x,
        "odds_2": odds_2,
        "odds_btts": odds_btts,
        "p_home": round(ris_co["p_home"]*100, 2),
        "p_draw": round(ris_co["p_draw"]*100, 2),
        "p_away": round(ris_co["p_away"]*100, 2),
        "btts": round(ris_co["btts"]*100, 2),
        "over_25": round(ris_co["over_25"]*100, 2),
        "gg_over25": round(ris_co["gg_over25"]*100, 2),
        "scost_1": round(ris_co["scost"]["1"], 2),
        "scost_x": round(ris_co["scost"]["X"], 2),
        "scost_2": round(ris_co["scost"]["2"], 2),
        "affidabilita": aff,
        "confidence": confidence
    }

    try:
        if os.path.exists(ARCHIVE_FILE):
            df_old = pd.read_csv(ARCHIVE_FILE)
            df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
            df_new.to_csv(ARCHIVE_FILE, index=False)
        else:
            pd.DataFrame([row]).to_csv(ARCHIVE_FILE, index=False)
        st.success("📁 Analisi salvata in storico_analisi.csv")
    except Exception as e:
        st.warning(f"Non sono riuscito a salvare l'analisi: {e}")

# ============================================================
#           ARCHIVIO MOSTRA
# ============================================================

st.subheader("📁 Archivio storico analisi")
if os.path.exists(ARCHIVE_FILE):
    st.dataframe(pd.read_csv(ARCHIVE_FILE).tail(50))
else:
    st.info("Nessun archivio trovato.")