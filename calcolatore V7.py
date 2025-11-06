import math
from typing import Dict, Any, List, Tuple
from datetime import datetime, date, timezone, timedelta
import pandas as pd
import os
import requests
import streamlit as st

# ============================================================
#                 CONFIG
# ============================================================

# The Odds API (PRO)
THE_ODDS_API_KEY = "06c16ede44d09f9b3498bb63354930c4"
THE_ODDS_BASE = "https://api.the-odds-api.com/v4"

# API-FOOTBALL solo per aggiornare lo storico
API_FOOTBALL_KEY = "95c43f936816cd4389a747fd2cfe061a"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ARCHIVE_FILE = "storico_analisi.csv"

# book affidabili da mediare (metterà la media di quelli trovati)
PREFERRED_BOOKS = [
    "pinnacle",
    "betonlineag",
    "bet365",
    "unibet_eu",
    "williamhill"
]

# ============================================================
#         FUNZIONI THE ODDS API (per scegliere la partita)
# ============================================================

def oddsapi_get_soccer_leagues() -> List[dict]:
    """
    Prende la lista di tutti gli sport e filtra solo quelli di calcio.
    Con il PRO dovresti vedere tutti i soccer_* disponibili.
    """
    try:
        r = requests.get(
            f"{THE_ODDS_BASE}/sports",
            params={"apiKey": THE_ODDS_API_KEY, "all": "true"},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        # filtro solo calcio
        soccer = [s for s in data if s.get("key", "").startswith("soccer")]
        return soccer
    except Exception as e:
        print("errore sports:", e)
        return []


def oddsapi_get_events_for_league(league_key: str) -> List[dict]:
    """
    Prende gli eventi di OGGI per una lega di calcio.
    """
    try:
        r = requests.get(
            f"{THE_ODDS_BASE}/sports/{league_key}/odds",
            params={
                "apiKey": THE_ODDS_API_KEY,
                "regions": "eu,uk",
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
                "dateFormat": "iso"
            },
            timeout=8
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("errore events:", e)
        return []


def oddsapi_extract_prices(event: dict) -> dict:
    """
    Da 1 evento della Odds API estrae la media di 1,X,2 e over/under 2.5.
    Restituisce un dict compatibile col nostro modello.
    """
    out = {
        "home": event.get("home_team"),
        "away": event.get("away_team"),
        "odds_1": None,
        "odds_x": None,
        "odds_2": None,
        "odds_over25": None,
        "odds_under25": None,
    }

    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return out

    # accumulatori per media
    h2h_home, h2h_draw, h2h_away = [], [], []
    over25_list, under25_list = [], []

    for bk in bookmakers:
        bk_key = bk.get("key")
        if bk_key not in PREFERRED_BOOKS:
            continue

        for mk in bk.get("markets", []):
            mk_key = mk.get("key")
            # 1X2
            if mk_key == "h2h":
                outcomes = mk.get("outcomes", [])
                # possono essere 2 o 3 (pareggio non sempre c'è)
                for o in outcomes:
                    name = o.get("name", "")
                    price = o.get("price")
                    if not price:
                        continue
                    if name == out["home"]:
                        h2h_home.append(price)
                    elif name == out["away"]:
                        h2h_away.append(price)
                    elif name.lower() in ["draw", "tie", "x"]:
                        h2h_draw.append(price)

            # totali
            elif mk_key == "totals":
                # cerco la linea 2.5
                for o in mk.get("outcomes", []):
                    point = o.get("point")
                    price = o.get("price")
                    name = o.get("name", "").lower()
                    if point == 2.5 and price:
                        if "over" in name:
                            over25_list.append(price)
                        elif "under" in name:
                            under25_list.append(price)

    # faccio le medie
    if h2h_home:
        out["odds_1"] = sum(h2h_home) / len(h2h_home)
    if h2h_draw:
        out["odds_x"] = sum(h2h_draw) / len(h2h_draw)
    if h2h_away:
        out["odds_2"] = sum(h2h_away) / len(h2h_away)
    if over25_list:
        out["odds_over25"] = sum(over25_list) / len(over25_list)
    if under25_list:
        out["odds_under25"] = sum(under25_list) / len(under25_list)

    return out

# ============================================================
#            API-FOOTBALL SOLO PER RISULTATI REALI
# ============================================================

def apifootball_get_fixtures_by_date(d: str) -> list:
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"date": d}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception:
        return []

# ============================================================
#                  FUNZIONI MODELLO (TUO BLOCCO)
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

    # non sempre abbiamo BTTS dalla Odds API → se non c'è usiamo fallback
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
#              STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Modello Scommesse – Odds API PRO", layout="wide")
st.title("⚽ Modello Scommesse – versione con The Odds API PRO")

st.caption(f"Esecuzione: {datetime.now().isoformat(timespec='seconds')}")

# init session state
if "soccer_leagues" not in st.session_state:
    st.session_state.soccer_leagues = []
if "events_for_league" not in st.session_state:
    st.session_state.events_for_league = []
if "selected_event_prices" not in st.session_state:
    st.session_state.selected_event_prices = {}

# ============================================================
#               SEZIONE STORICO
# ============================================================

st.subheader("📁 Stato storico")
if os.path.exists(ARCHIVE_FILE):
    df_st = pd.read_csv(ARCHIVE_FILE)
    st.write(f"Analisi salvate: **{len(df_st)}**")
    st.dataframe(df_st.tail(30))
else:
    st.info("Nessuno storico ancora.")

st.markdown("---")

# ============================================================
# 0. PRENDI PARTITA DALLA ODDS API
# ============================================================

st.subheader("🔍 Prendi partita dall’API e riempi i campi")

col_a, col_b = st.columns([1,2])

with col_a:
    if st.button("1) Carica leghe calcio dalla Odds API"):
        st.session_state.soccer_leagues = oddsapi_get_soccer_leagues()
        st.session_state.events_for_league = []
        st.success(f"Trovate {len(st.session_state.soccer_leagues)} leghe calcio.")

if st.session_state.soccer_leagues:
    league_names = [f"{l['title']} ({l['key']})" for l in st.session_state.soccer_leagues]
    selected_league_label = st.selectbox("2) Seleziona la lega", league_names)
    # ricavo la key vera
    selected_league_key = selected_league_label.split("(")[-1].replace(")", "").strip()

    if st.button("3) Carica partite di questa lega"):
        st.session_state.events_for_league = oddsapi_get_events_for_league(selected_league_key)
        st.success(f"Partite trovate: {len(st.session_state.events_for_league)}")

    if st.session_state.events_for_league:
        # creo le label per il select
        match_labels = []
        for ev in st.session_state.events_for_league:
            home = ev.get("home_team")
            away = ev.get("away_team")
            start = ev.get("commence_time", "")[:16].replace("T", " ")
            match_labels.append(f"{home} vs {away} – {start}")
        selected_match_label = st.selectbox("4) Seleziona la partita", match_labels)

        # quando scegli la partita, estrai le quote
        idx = match_labels.index(selected_match_label)
        event = st.session_state.events_for_league[idx]
        prices = oddsapi_extract_prices(event)
        st.session_state.selected_event_prices = prices
        st.success("Quote prese dall’API e pronte per essere usate sotto ✅")

# ============================================================
# 1. DATI PARTITA
# ============================================================

st.subheader("1. Dati partita")

default_match_name = ""
if st.session_state.get("selected_event_prices", {}).get("home"):
    default_match_name = f"{st.session_state['selected_event_prices']['home']} vs {st.session_state['selected_event_prices']['away']}"

match_name = st.text_input("Nome partita (es. Milan vs Inter)", value=default_match_name)

# ============================================================
# 2. LINEE DI APERTURA (sempre manuali)
# ============================================================

st.subheader("2. Linee di apertura (manuali)")
col_ap1, col_ap2 = st.columns(2)
with col_ap1:
    spread_ap = st.number_input("Spread apertura", value=0.0, step=0.25)
with col_ap2:
    total_ap = st.number_input("Total apertura", value=2.5, step=0.25)

# ============================================================
# 3. LINEE CORRENTI E QUOTE (precompilate se trovate)
# ============================================================

st.subheader("3. Linee correnti e quote (manuali ma precompilate se trovate)")

api_prices = st.session_state.get("selected_event_prices", {})

col_co1, col_co2, col_co3 = st.columns(3)
with col_co1:
    spread_co = st.number_input("Spread corrente", value=0.0, step=0.25)
    odds_1 = st.number_input("Quota 1", value=float(api_prices.get("odds_1") or 1.80), step=0.01)
with col_co2:
    total_co = st.number_input("Total corrente", value=2.5, step=0.25)
    odds_x = st.number_input("Quota X", value=float(api_prices.get("odds_x") or 3.50), step=0.01)
with col_co3:
    odds_2 = st.number_input("Quota 2", value=float(api_prices.get("odds_2") or 4.50), step=0.01)
    odds_btts = st.number_input("Quota GG (BTTS sì) – se ce l’hai tu", value=1.95, step=0.01)

st.subheader("3.b Quote Over/Under 2.5 (dall’API se c’è)")
col_ou1, col_ou2 = st.columns(2)
with col_ou1:
    odds_over25 = st.number_input("Quota Over 2.5", value=float(api_prices.get("odds_over25") or 0.0), step=0.01)
with col_ou2:
    odds_under25 = st.number_input("Quota Under 2.5", value=float(api_prices.get("odds_under25") or 0.0), step=0.01)

# ============================================================
# 4. XG (sempre manuali)
# ============================================================

st.subheader("4. xG avanzati (opzionali)")
col_xg1, col_xg2 = st.columns(2)
with col_xg1:
    xg_tot_home = st.text_input("xG totali CASA", "")
    xga_tot_home = st.text_input("xGA totali CASA", "")
    partite_home = st.text_input("Partite giocate CASA (es. 10 o 5-3-2)", "")
with col_xg2:
    xg_tot_away = st.text_input("xG totali OSPITE", "")
    xga_tot_away = st.text_input("xGA totali OSPITE", "")
    partite_away = st.text_input("Partite giocate OSPITE (es. 10 o 5-3-2)", "")

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
    st.info("Modalità: BASE (spread/total/quote). Se inserisci xG passo in modalità avanzata.")
else:
    st.success("Modalità: AVANZATA (spread/total + quote + xG/xGA).")

# ============================================================
# 5. CALCOLO MODELLO
# ============================================================

if st.button("CALCOLA MODELLO"):
    # due calcoli: apertura e corrente
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

    # calcolo affidabilità semplice
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
    aff = max(0, min(100, aff))

    st.success("Calcolo completato ✅")
    st.subheader("⭐ Sintesi Match")
    st.write(f"Affidabilità del match: **{aff}/100**")

    # movimento
    delta_spread = spread_co - spread_ap
    delta_total = total_co - total_ap

    st.subheader("🔁 Movimento di mercato")
    if abs(delta_spread) < 0.01 and abs(delta_total) < 0.01:
        st.write("Linee stabili.")
    else:
        if abs(delta_spread) >= 0.01:
            if delta_spread < 0:
                st.write(f"- Spread sceso di {abs(delta_spread):.2f} → mercato più pro CASA")
            else:
                st.write(f"- Spread salito di {abs(delta_spread):.2f} → mercato più pro TRASFERTA")
        if abs(delta_total) >= 0.01:
            if delta_total > 0:
                st.write(f"- Total salito di {delta_total:.2f} → mercato si aspetta più gol")
            else:
                st.write(f"- Total sceso di {abs(delta_total):.2f} → mercato si aspetta meno gol")

    # value finder basilare
    st.subheader("💰 Value Finder")
    rows = []
    for lab, p_mod, odd in [
        ("1", ris_co["p_home"], odds_1),
        ("X", ris_co["p_draw"], odds_x),
        ("2", ris_co["p_away"], odds_2),
    ]:
        p_book = decimali_a_prob(odd)
        diff = (p_mod - p_book) * 100
        rows.append({
            "Mercato": "1X2",
            "Esito": lab,
            "Prob modello %": round(p_mod*100, 2),
            "Prob quota %": round(p_book*100, 2),
            "Δ pp": round(diff, 2),
        })

    if odds_over25 and odds_over25 > 1:
        p_mod = ris_co["over_25"]
        p_book = decimali_a_prob(odds_over25)
        diff = (p_mod - p_book) * 100
        rows.append({
            "Mercato": "Over/Under 2.5",
            "Esito": "Over 2.5",
            "Prob modello %": round(p_mod*100, 2),
            "Prob quota %": round(p_book*100, 2),
            "Δ pp": round(diff, 2),
        })

    if odds_under25 and odds_under25 > 1:
        p_mod = ris_co["under_25"]
        p_book = decimali_a_prob(odds_under25)
        diff = (p_mod - p_book) * 100
        rows.append({
            "Mercato": "Over/Under 2.5",
            "Esito": "Under 2.5",
            "Prob modello %": round(p_mod*100, 2),
            "Prob quota %": round(p_book*100, 2),
            "Δ pp": round(diff, 2),
        })

    st.dataframe(pd.DataFrame(rows))

    # espansioni
    with st.expander("Top 10 Risultati esatti"):
        for h, a, p in ris_co["top10"]:
            st.write(f"{h}-{a}: {p:.1f}%")

    # salvataggio
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
        "odds_over25": odds_over25,
        "odds_under25": odds_under25,
        "p_home": round(ris_co["p_home"]*100, 2),
        "p_draw": round(ris_co["p_draw"]*100, 2),
        "p_away": round(ris_co["p_away"]*100, 2),
        "btts": round(ris_co["btts"]*100, 2),
        "over_25": round(ris_co["over_25"]*100, 2),
        "affidabilita": aff,
        "esito_modello": max(
            [("1", ris_co["p_home"]), ("X", ris_co["p_draw"]), ("2", ris_co["p_away"])],
            key=lambda x: x[1]
        )[0],
        "esito_reale": "",
        "risultato_reale": "",
        "match_ok": ""
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
#           AGGIORNA RISULTATI REALI (API-FOOTBALL)
# ============================================================
st.subheader("🔄 Aggiorna risultati reali nello storico (API-Football)")

if st.button("Recupera risultati degli ultimi 3 giorni"):
    if not os.path.exists(ARCHIVE_FILE):
        st.warning("Non c'è ancora uno storico da aggiornare.")
    else:
        df = pd.read_csv(ARCHIVE_FILE)
        today = date.today()
        giorni_da_controllare = [(today - timedelta(days=i)).isoformat() for i in range(0, 4)]
        fixtures_by_day = {}
        for d in giorni_da_controllare:
            fixtures_by_day[d] = apifootball_get_fixtures_by_date(d)

        results_map = {}
        for d, fixtures in fixtures_by_day.items():
            for f in fixtures:
                if f["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]:
                    home = f["teams"]["home"]["name"]
                    away = f["teams"]["away"]["name"]
                    key = f"{home} vs {away}".strip().lower()
                    goals_home = f["goals"]["home"]
                    goals_away = f["goals"]["away"]
                    results_map[key] = (goals_home, goals_away)

        updated = 0
        for idx, row in df.iterrows():
            key_row = str(row.get("match", "")).strip().lower()
            if key_row in results_map and (pd.isna(row.get("risultato_reale")) or row.get("risultato_reale") == ""):
                gh, ga = results_map[key_row]
                if gh is None or ga is None:
                    continue
                if gh > ga:
                    esito_real = "1"
                elif gh == ga:
                    esito_real = "X"
                else:
                    esito_real = "2"
                df.at[idx, "risultato_reale"] = f"{gh}-{ga}"
                df.at[idx, "esito_reale"] = esito_real
                pred = row.get("esito_modello", "")
                if pred != "" and esito_real != "":
                    df.at[idx, "match_ok"] = 1 if pred == esito_real else 0
                updated += 1

        df.to_csv(ARCHIVE_FILE, index=False)
        st.success(f"Aggiornamento completato. Partite aggiornate: {updated}")
        st.dataframe(df.tail(30))