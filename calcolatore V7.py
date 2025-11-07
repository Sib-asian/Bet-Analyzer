import math
from typing import Dict, Any, List, Tuple
from datetime import datetime, date, timedelta
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

# API-FOOTBALL solo per aggiornare lo storico (risultati reali)
API_FOOTBALL_KEY = "95c43f936816cd4389a747fd2cfe061a"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ARCHIVE_FILE = "storico_analisi.csv"

# book affidabili da mediare
PREFERRED_BOOKS = [
    "pinnacle",
    "betonlineag",
    "bet365",
    "unibet_eu",
    "williamhill",
]

# ============================================================
#         FUNZIONI THE ODDS API (per scegliere la partita)
# ============================================================

def oddsapi_get_soccer_leagues() -> List[dict]:
    """Prende tutti gli sport e filtra solo quelli di calcio."""
    try:
        r = requests.get(
            f"{THE_ODDS_BASE}/sports",
            params={"apiKey": THE_ODDS_API_KEY, "all": "true"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        soccer = [s for s in data if s.get("key", "").startswith("soccer")]
        return soccer
    except Exception as e:
        print("errore sports:", e)
        return []


def oddsapi_get_events_for_league(league_key: str) -> List[dict]:
    """
    Prende gli eventi per una lega di calcio.
    markets: h2h, totals, spreads, btts → per avere 1X2, over/under, DNB e GG.
    """
    try:
        r = requests.get(
            f"{THE_ODDS_BASE}/sports/{league_key}/odds",
            params={
                "apiKey": THE_ODDS_API_KEY,
                "regions": "eu,uk",
                "markets": "h2h,totals,spreads,btts",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=8,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("errore events:", e)
        return []


def oddsapi_extract_prices(event: dict) -> dict:
    """
    Da 1 evento della Odds API estrae:
    - media 1X2
    - media Over/Under 2.5
    - DNB Casa / DNB Trasferta ricavati dallo spread 0
    - quota GG (BTTS yes)
    """
    out = {
        "home": event.get("home_team"),
        "away": event.get("away_team"),
        "odds_1": None,
        "odds_x": None,
        "odds_2": None,
        "odds_over25": None,
        "odds_under25": None,
        "odds_dnb_home": None,
        "odds_dnb_away": None,
        "odds_btts": None,
    }

    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return out

    # accumulatori per medie
    h2h_home, h2h_draw, h2h_away = [], [], []
    over25_list, under25_list = [], []
    dnb_home_list, dnb_away_list = [], []
    btts_list = []

    for bk in bookmakers:
        bk_key = bk.get("key")
        if bk_key not in PREFERRED_BOOKS:
            continue

        for mk in bk.get("markets", []):
            mk_key = mk.get("key")

            # 1X2
            if mk_key == "h2h":
                for o in mk.get("outcomes", []):
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

            # totals → cerchiamo la linea 2.5
            elif mk_key == "totals":
                for o in mk.get("outcomes", []):
                    point = o.get("point")
                    price = o.get("price")
                    name = o.get("name", "").lower()
                    if point == 2.5 and price:
                        if "over" in name:
                            over25_list.append(price)
                        elif "under" in name:
                            under25_list.append(price)

            # spreads → se il point è 0 lo leggiamo come DNB
            elif mk_key == "spreads":
                for o in mk.get("outcomes", []):
                    point = o.get("point")
                    price = o.get("price")
                    name = o.get("name", "")
                    if price is None:
                        continue
                    if point == 0:
                        if name == out["home"]:
                            dnb_home_list.append(price)
                        elif name == out["away"]:
                            dnb_away_list.append(price)

            # BTTS / entrambe segnano
            elif mk_key in ("btts", "both_teams_to_score"):
                for o in mk.get("outcomes", []):
                    name = o.get("name", "").lower()
                    price = o.get("price")
                    if not price:
                        continue
                    # di solito "Yes"
                    if "yes" in name or "sì" in name or "si" in name:
                        btts_list.append(price)

    # medie
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
    if dnb_home_list:
        out["odds_dnb_home"] = sum(dnb_home_list) / len(dnb_home_list)
    if dnb_away_list:
        out["odds_dnb_away"] = sum(dnb_away_list) / len(dnb_away_list)
    if btts_list:
        out["odds_btts"] = sum(btts_list) / len(btts_list)

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
    except Exception as e:
        print("errore api-football:", e)
        return []

# ============================================================
#                  FUNZIONI MODELLO
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

def risultato_completo(
    spread: float,
    total: float,
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_btts: float,
    xg_for_home: float = None,
    xg_against_home: float = None,
    xg_for_away: float = None,
    xg_against_away: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
) -> Dict[str, Any]:

    # 1) base da 1X2
    p1, px, p2 = normalize_1x2_from_odds(odds_1, odds_x, odds_2)

    # 2) se abbiamo DNB li traduciamo in probabilità e li fondiamo
    if odds_dnb_home and odds_dnb_home > 1 and odds_dnb_away and odds_dnb_away > 1:
        pdnb_home = 1 / odds_dnb_home
        pdnb_away = 1 / odds_dnb_away
        tot_dnb = pdnb_home + pdnb_away
        if tot_dnb > 0:
            pdnb_home /= tot_dnb
            pdnb_away /= tot_dnb
            p1 = p1 * 0.7 + pdnb_home * 0.3
            p2 = p2 * 0.7 + pdnb_away * 0.3
            px = max(0.0, 1.0 - (p1 + p2))

    lh, la = gol_attesi_migliorati(spread, total, p1, p2)

    if (xg_for_home is not None and xg_against_home is not None and
        xg_for_away is not None and xg_against_away is not None):
        lh, la = blend_lambda_market_xg(
            lh, la,
            xg_for_home, xg_against_home,
            xg_for_away, xg_against_away,
            w_market=0.6
        )

    # rho (correlazione BTTS)
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
        "scost": scost,
    }

# ============================================================
#   NUOVE FUNZIONI: check coerenza, market pressure, confidence
# ============================================================

def check_coerenza_quote(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float,
    odds_under25: float,
) -> List[str]:
    """Controlli veloci per vedere se le quote hanno qualcosa di storto."""
    warnings = []

    # 1x2 base
    if odds_1 and odds_2 and odds_1 < 1.25 and odds_2 < 5:
        warnings.append("Casa troppo favorita ma trasferta non abbastanza alta.")
    if odds_1 and odds_2 and odds_1 > 3.0 and odds_2 > 3.0:
        warnings.append("Sia casa che trasferta sopra 3.0 → match molto caotico.")

    # over/under
    if odds_over25 and odds_under25:
        p_over = 1 / odds_over25
        p_under = 1 / odds_under25
        somma = p_over + p_under

        # range realistico per OU 2.5 con margine
        if not (1.00 < somma < 1.25):
            warnings.append("Mercato over/under 2.5 con margine anomalo (controlla le quote).")

        # se 1 è super favorita ma over è alto
        if odds_1 and odds_1 < 1.5 and odds_over25 > 2.2:
            warnings.append("Favorita netta ma over 2.5 alto → controlla linea gol.")
    else:
        warnings.append("Manca almeno una quota Over/Under 2.5 → controlli incompleti.")

    return warnings

def compute_market_pressure_index(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float,
    odds_under25: float,
    odds_dnb_home: float,
    odds_dnb_away: float,
) -> int:
    """
    0–100: più alto = mercato pulito e direzionale.
    """
    score = 50  # base

    # favorito chiaro
    if odds_1 and odds_2:
        if odds_1 < 1.7 and odds_2 > 3.5:
            score += 20
        elif odds_1 < 2.0 and odds_2 > 3.0:
            score += 10

    # dnb che conferma
    if odds_dnb_home and odds_dnb_home > 1:
        if odds_1 and odds_1 < 2.0 and odds_dnb_home < 1.5:
            score += 10
    if odds_dnb_away and odds_dnb_away > 1:
        if odds_2 and odds_2 < 2.0 and odds_dnb_away < 1.5:
            score += 10

    # over/under ragionevoli
    if odds_over25 and odds_under25:
        p_over = 1 / odds_over25
        p_under = 1 / odds_under25
        somma = p_over + p_under
        if 1.00 < somma < 1.25:
            score += 5
        else:
            score -= 5
    else:
        score -= 5

    return max(0, min(100, score))

def compute_structure_affidability(
    spread_ap: float,
    spread_co: float,
    total_ap: float,
    total_co: float,
    ent_media: float,
    has_xg: bool,
    odds_1: float,
    odds_x: float,
    odds_2: float
) -> int:
    """
    Affidabilità strutturale del match: più variazioni ci sono, più scende.
    """
    aff = 100

    diff_spread = abs(spread_ap - spread_co)
    diff_total = abs(total_ap - total_co)

    # ogni 0.25 di differenza toglie
    aff -= int(diff_spread / 0.25) * 8
    aff -= int(diff_total / 0.25) * 5

    # entropia alta → match sporco
    if ent_media > 2.25:
        aff -= 15
    elif ent_media > 2.10:
        aff -= 8

    # niente xG → un po' meno affidabile
    if not has_xg:
        aff -= 7

    # 1X2 troppo piatta → meno affidabile
    if odds_1 and odds_x and odds_2:
        probs = [1/odds_1, 1/odds_x, 1/odds_2]
        spread_prob = max(probs) - min(probs)
        if spread_prob < 0.10:
            aff -= 8

    return max(0, min(100, aff))

def compute_global_confidence(
    base_aff: int,
    n_warnings: int,
    mpi: int,
    has_xg: bool,
) -> int:
    """
    mix: affidabilità tua, penalità per warning, bonus per market pressure, bonus se hai xG
    """
    conf = base_aff
    conf -= n_warnings * 5
    conf += int((mpi - 50) * 0.3)  # se mpi > 50 aggiunge, se < 50 toglie
    if has_xg:
        conf += 5
    return max(0, min(100, conf))

# ============================================================
#              STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Modello Scommesse – Odds API PRO", layout="wide")
st.title("⚽ Modello Scommesse – versione con The Odds API PRO + DNB + controlli")

st.caption(f"Esecuzione: {datetime.now().isoformat(timespec='seconds')}")

# init session state
if "soccer_leagues" not in st.session_state:
    st.session_state.soccer_leagues = []
if "events_for_league" not in st.session_state:
    st.session_state.events_for_league = []
if "selected_event_prices" not in st.session_state:
    st.session_state.selected_event_prices = {}

# ============================================================
#               SEZIONE STORICO + CANCELLA
# ============================================================

st.subheader("📁 Stato storico")
if os.path.exists(ARCHIVE_FILE):
    df_st = pd.read_csv(ARCHIVE_FILE)
    st.write(f"Analisi salvate: **{len(df_st)}**")
    st.dataframe(df_st.tail(30))
else:
    st.info("Nessuno storico ancora.")

st.markdown("### 🗑️ Cancella analisi dallo storico")
if os.path.exists(ARCHIVE_FILE):
    df_del = pd.read_csv(ARCHIVE_FILE)
    if not df_del.empty:
        df_del["label"] = df_del.apply(
            lambda r: f"{r.get('timestamp','?')} – {r.get('match','(senza nome)')}",
            axis=1,
        )
        to_delete = st.selectbox(
            "Seleziona la riga da eliminare:",
            df_del["label"].tolist()
        )
        if st.button("Elimina riga selezionata"):
            df_new = df_del[df_del["label"] != to_delete].drop(columns=["label"])
            df_new.to_csv(ARCHIVE_FILE, index=False)
            st.success("✅ Riga eliminata. Ricarica la pagina per vedere l’archivio aggiornato.")
    else:
        st.info("Lo storico è vuoto, niente da cancellare.")
else:
    st.info("Nessun file storico, niente da cancellare.")

st.markdown("---")

# ============================================================
# 0. PRENDI PARTITA DALL’API
# ============================================================

st.subheader("🔍 Prendi una partita da The Odds API e riempi le quote")

col_a, col_b = st.columns([1, 2])

with col_a:
    if st.button("1) Carica leghe di calcio"):
        st.session_state.soccer_leagues = oddsapi_get_soccer_leagues()
        st.session_state.events_for_league = []
        if st.session_state.soccer_leagues:
            st.success(f"Trovate {len(st.session_state.soccer_leagues)} leghe calcio.")
        else:
            st.warning("Non sono riuscito a caricare le leghe. Controlla API key / limiti.")

if st.session_state.soccer_leagues:
    league_names = [f"{l['title']} ({l['key']})" for l in st.session_state.soccer_leagues]
    selected_league_label = st.selectbox("2) Seleziona la lega", league_names)
    selected_league_key = selected_league_label.split("(")[-1].replace(")", "").strip()

    if st.button("3) Carica partite di questa lega"):
        st.session_state.events_for_league = oddsapi_get_events_for_league(selected_league_key)
        st.success(f"Partite trovate: {len(st.session_state.events_for_league)}")

    if st.session_state.events_for_league:
        match_labels = []
        for ev in st.session_state.events_for_league:
            home = ev.get("home_team")
            away = ev.get("away_team")
            start = ev.get("commence_time", "")[:16].replace("T", " ")
            match_labels.append(f"{home} vs {away} – {start}")

        selected_match_label = st.selectbox("4) Seleziona la partita", match_labels)
        idx = match_labels.index(selected_match_label)
        event = st.session_state.events_for_league[idx]
        prices = oddsapi_extract_prices(event)
        st.session_state.selected_event_prices = prices
        st.success("Quote prese dall’API e precompilate più sotto ✅")

# ============================================================
# 1. DATI PARTITA
# ============================================================

st.subheader("1. Dati partita")

default_match_name = ""
if st.session_state.get("selected_event_prices", {}).get("home"):
    default_match_name = f"{st.session_state['selected_event_prices']['home']} vs {st.session_state['selected_event_prices']['away']}"

match_name = st.text_input("Nome partita (es. Milan vs Inter)", value=default_match_name)

# ============================================================
# 2. LINEE DI APERTURA
# ============================================================

st.subheader("2. Linee di apertura (manuali)")
col_ap1, col_ap2 = st.columns(2)
with col_ap1:
    spread_ap = st.number_input("Spread apertura", value=0.0, step=0.25)
with col_ap2:
    total_ap = st.number_input("Total apertura", value=2.5, step=0.25)

# ============================================================
# 3. LINEE CORRENTI E QUOTE
# ============================================================

st.subheader("3. Linee correnti e quote (precompilate)")

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
    # GG ora precompilato dall’API se c’è
    odds_btts = st.number_input(
        "Quota GG (BTTS sì)",
        value=float(api_prices.get("odds_btts") or 1.95),
        step=0.01
    )

# DNB precompilati
st.subheader("3.b DNB (Draw No Bet) – letti dallo spread 0 se disponibili")
col_dnb1, col_dnb2 = st.columns(2)
with col_dnb1:
    odds_dnb_home = st.number_input("Quota DNB Casa", value=float(api_prices.get("odds_dnb_home") or 0.0), step=0.01)
with col_dnb2:
    odds_dnb_away = st.number_input("Quota DNB Trasferta", value=float(api_prices.get("odds_dnb_away") or 0.0), step=0.01)

# Over / Under
st.subheader("3.c Quote Over/Under 2.5")
col_ou1, col_ou2 = st.columns(2)
with col_ou1:
    odds_over25 = st.number_input("Quota Over 2.5", value=float(api_prices.get("odds_over25") or 0.0), step=0.01)
with col_ou2:
    odds_under25 = st.number_input("Quota Under 2.5", value=float(api_prices.get("odds_under25") or 0.0), step=0.01)

# ============================================================
# 4. XG (manuali)
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
    # calcolo apertura
    ris_ap = risultato_completo(
        spread_ap, total_ap,
        odds_1, odds_x, odds_2,
        0.0,  # niente btts apertura
        xg_home_for, xg_home_against,
        xg_away_for, xg_away_against,
        odds_dnb_home=odds_dnb_home if odds_dnb_home > 0 else None,
        odds_dnb_away=odds_dnb_away if odds_dnb_away > 0 else None,
    )
    # calcolo corrente
    ris_co = risultato_completo(
        spread_co, total_co,
        odds_1, odds_x, odds_2,
        odds_btts,
        xg_home_for, xg_home_against,
        xg_away_for, xg_away_against,
        odds_dnb_home=odds_dnb_home if odds_dnb_home > 0 else None,
        odds_dnb_away=odds_dnb_away if odds_dnb_away > 0 else None,
    )

    ent_media = (ris_co["ent_home"] + ris_co["ent_away"]) / 2

    # 1) check coerenza quote
    warnings = check_coerenza_quote(
        odds_1, odds_x, odds_2,
        odds_over25, odds_under25
    )

    # 2) market pressure index
    mpi = compute_market_pressure_index(
        odds_1, odds_x, odds_2,
        odds_over25, odds_under25,
        odds_dnb_home, odds_dnb_away
    )

    # 3) affidabilità della struttura
    aff = compute_structure_affidability(
        spread_ap, spread_co,
        total_ap, total_co,
        ent_media,
        has_xg,
        odds_1, odds_x, odds_2
    )

    # 4) confidence globale
    global_conf = compute_global_confidence(
        base_aff=aff,
        n_warnings=len(warnings),
        mpi=mpi,
        has_xg=has_xg
    )

    st.success("Calcolo completato ✅")
    st.subheader("⭐ Sintesi Match")
    st.write(f"Affidabilità del match (struttura): **{aff}/100**")
    st.write(f"Confidence globale: **{global_conf}/100**")
    st.write(f"Market Pressure Index: **{mpi}/100**")

    if warnings:
        st.subheader("⚠️ Check coerenza quote")
        for w in warnings:
            st.write(f"- {w}")
    else:
        st.subheader("✅ Check coerenza quote")
        st.write("Quote coerenti con il modello minimo.")

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

    # value finder
    st.subheader("💰 Value Finder")
    rows = []

    # flag se OU è anomalo
    anomalo_ou = any("over/under 2.5" in w.lower() for w in warnings)

    # 1X2
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

    # Over/Under solo se quote sane
    if not anomalo_ou:
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

    # GG (BTTS) – ora arriva già dall’API
    if odds_btts and odds_btts > 1:
        p_mod = ris_co["btts"]
        p_book = decimali_a_prob(odds_btts)
        diff = (p_mod - p_book) * 100
        rows.append({
            "Mercato": "Entrambe segnano",
            "Esito": "Sì",
            "Prob modello %": round(p_mod*100, 2),
            "Prob quota %": round(p_book*100, 2),
            "Δ pp": round(diff, 2),
        })

    st.dataframe(pd.DataFrame(rows))

    # espansioni
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

    with st.expander("9️⃣ Combo mercati (1&Over, DC+GG, ecc.)"):
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

    # salvataggio nel CSV
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
        "odds_dnb_home": odds_dnb_home,
        "odds_dnb_away": odds_dnb_away,
        "odds_btts": odds_btts,
        "p_home": round(ris_co["p_home"]*100, 2),
        "p_draw": round(ris_co["p_draw"]*100, 2),
        "p_away": round(ris_co["p_away"]*100, 2),
        "btts": round(ris_co["btts"]*100, 2),
        "over_25": round(ris_co["over_25"]*100, 2),
        "affidabilita": aff,
        "confidence_globale": global_conf,
        "market_pressure_index": mpi,
        "esito_modello": max(
            [("1", ris_co["p_home"]), ("X", ris_co["p_draw"]), ("2", ris_co["p_away"])],
            key=lambda x: x[1]
        )[0],
        "esito_reale": "",
        "risultato_reale": "",
        "match_ok": "",
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