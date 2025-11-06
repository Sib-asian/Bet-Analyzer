import math
from typing import Dict, Any, List, Tuple
from datetime import datetime, date, timezone, timedelta
import pandas as pd
import os
import requests
import streamlit as st

# ============================================================
#           CONFIGURAZIONE PRINCIPALE (V7.0 - THE ODDS API)
# ============================================================

ODDS_API_KEY = "dd98fa8bb347a59f78867a4fe680e1e8"  # tua key The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
ARCHIVE_FILE = "storico_analisi.csv"

# bookmaker che proviamo a considerare per mediare le quote
PREFERRED_BOOKS = {
    "Pinnacle",
    "Bet365",
    "Unibet",
    "10Bet",
    "Betfair",
    "William Hill",
    "Bwin"
}

# sport di default
DEFAULT_SPORT_KEY = "soccer_italy_serie_a"

# ============================================================
#                 FUNZIONI API: THE ODDS API
# ============================================================

def oddsapi_get_sports() -> List[dict]:
    """Ritorna la lista degli sport disponibili su The Odds API."""
    try:
        r = requests.get(
            f"{ODDS_API_BASE}",
            params={"apiKey": ODDS_API_KEY},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def oddsapi_get_odds_for_sport(
    sport_key: str,
    regions: str = "eu,uk",
    markets: str = "h2h,totals,btts",
    odds_format: str = "decimal"
) -> List[dict]:
    """
    Ritorna le quote per uno sport/lega.
    Non c'è il filtro per data su The Odds API free, quindi prendiamo il palinsesto attuale/prossimo.
    """
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/{sport_key}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def _clean_decimal(value: Any) -> float:
    try:
        v = float(value)
        if v <= 1:
            return 0.0
        return v
    except Exception:
        return 0.0

def aggregate_odds_from_event(ev: dict) -> dict:
    """
    ev è una singola partita restituita da The Odds API.
    Qui facciamo:
    - media su più bookmaker (solo quelli scelti)
    - scartiamo quote sballate su BTTS e Over/Under
    ritorniamo un dict uniforme come nel modello.
    """
    out = {
        "home": ev.get("home_team", ""),
        "away": ev.get("away_team", ""),
        "odds_1": None,
        "odds_x": None,
        "odds_2": None,
        "odds_btts": None,
        "odds_over25": None,
        "odds_under25": None,
        "bookmakers_used": []
    }

    bms = ev.get("bookmakers", [])
    if not bms:
        return out

    # accumulatori
    h2h_vals_1 = []
    h2h_vals_x = []
    h2h_vals_2 = []
    btts_vals = []
    over25_vals = []
    under25_vals = []
    used_bm_names = []

    for bm in bms:
        name = bm.get("title", "")
        if name and name not in PREFERRED_BOOKS:
            # se vuoi prendere tutti i book, commenta questo continue
            continue
        markets = bm.get("markets", [])
        if not markets:
            continue
        used_bm_names.append(name)

        for m in markets:
            mk = m.get("key", "")
            outcomes = m.get("outcomes", [])
            # 1X2
            if mk == "h2h":
                # qui spesso ci sono solo 2 outcomes (no pareggio) in alcuni campionati
                # ma nel calcio di solito 3
                for o in outcomes:
                    odd_val = _clean_decimal(o.get("price", 0))
                    if odd_val == 0.0:
                        continue
                    if o.get("name", "").lower() in ["home", out["home"].lower()]:
                        h2h_vals_1.append(odd_val)
                    elif o.get("name", "").lower() in ["away", out["away"].lower()]:
                        h2h_vals_2.append(odd_val)
                    elif o.get("name", "").lower() in ["draw", "tie", "x"]:
                        h2h_vals_x.append(odd_val)

            # totals (dobbiamo filtrare over 2.5 e under 2.5)
            elif mk == "totals":
                # outcomes: [{"name": "Over", "price": 1.9, "point": 2.5}, ...]
                for o in outcomes:
                    point = o.get("point")
                    name_out = o.get("name", "").lower()
                    odd_val = _clean_decimal(o.get("price", 0))
                    if not point or odd_val == 0.0:
                        continue
                    # consideriamo solo la linea 2.5
                    if abs(float(point) - 2.5) < 0.001:
                        if name_out == "over" and 1.1 <= odd_val <= 3.0:
                            over25_vals.append(odd_val)
                        elif name_out == "under" and 1.1 <= odd_val <= 3.0:
                            under25_vals.append(odd_val)

            # btts (some providers use "btts", some "both_teams_to_score")
            elif mk in ["btts", "both_teams_to_score"]:
                for o in outcomes:
                    name_out = o.get("name", "").lower()
                    odd_val = _clean_decimal(o.get("price", 0))
                    # vogliamo solo il "yes"
                    if name_out in ["yes", "sì", "si"] and 1.1 <= odd_val <= 3.0:
                        btts_vals.append(odd_val)

    # ora facciamo le medie
    def _avg(xs: list) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    out["odds_1"] = _avg(h2h_vals_1) if h2h_vals_1 else None
    out["odds_x"] = _avg(h2h_vals_x) if h2h_vals_x else None
    out["odds_2"] = _avg(h2h_vals_2) if h2h_vals_2 else None
    out["odds_btts"] = _avg(btts_vals) if btts_vals else None
    out["odds_over25"] = _avg(over25_vals) if over25_vals else None
    out["odds_under25"] = _avg(under25_vals) if under25_vals else None
    out["bookmakers_used"] = used_bm_names

    return out

# ============================================================
#                  FUNZIONI MODELLO (COME V6)
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
#        CHECK DI AFFIDABILITÀ + CONFIDENCE ENGINE
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

    low_scoring = ["serie b", "lige 1", "liga 2", "brazil serie b", "segunda"]
    if league_name and league_name.lower() in low_scoring:
        penalty += 3

    if fixture_datetime:
        now_utc = datetime.now(timezone.utc)
        diff_days = (fixture_datetime - now_utc).days
        if diff_days >= 2:
            penalty += 5
        elif diff_days == 1:
            penalty += 2

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
#                   STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Modello Scommesse V7.0 – The Odds API", layout="wide")
st.title("📊 Modello Scommesse V7.0 – Manuale + The Odds API + Scanner")

st.caption(f"Esecuzione: {datetime.now().isoformat(timespec='seconds')}")

# session init
if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = None
if "api_events" not in st.session_state:
    st.session_state.api_events = []
if "selected_sport_key" not in st.session_state:
    st.session_state.selected_sport_key = DEFAULT_SPORT_KEY

# ============================================================
#               SEZIONE 0: STORICO & PERFORMANCE
# ============================================================

st.subheader("📁 Storico & performance")

if os.path.exists(ARCHIVE_FILE):
    df_st = pd.read_csv(ARCHIVE_FILE)
    st.write(f"📦 Partite salvate: **{len(df_st)}**")
    if "match_ok" in df_st.columns and df_st["match_ok"].notna().any():
        df_val = df_st[df_st["match_ok"].isin([0, 1])]
        if len(df_val) > 0:
            acc = df_val["match_ok"].mean() * 100
            st.write(f"🎯 Accuracy (partite aggiornate): **{acc:.1f}%**")
else:
    st.info("Nessuno storico ancora. Calcola una partita per popolare il CSV.")

st.markdown("---")

# ============================================================
#       SEZIONE API – SCELTA LEGA (THE ODDS API)
# ============================================================

st.subheader("0. Seleziona lega/sport da The Odds API")

sports_list = oddsapi_get_sports()
# filtriamo solo soccer per non fare casino
soccer_sports = [s for s in sports_list if s.get("group", "").lower().startswith("soccer")]

if soccer_sports:
    # facciamo un dict nome → key
    sport_labels = {f"{s.get('details','')} ({s.get('title','')})": s.get("key") for s in soccer_sports}
    # se il default esiste, lo mettiamo
    keys_list = list(sport_labels.keys())
    if st.session_state.selected_sport_key in sport_labels.values():
        # troviamo la label
        default_index = keys_list.index(
            [lbl for lbl, key in sport_labels.items() if key == st.session_state.selected_sport_key][0]
        )
    else:
        default_index = 0
    selected_sport_label = st.selectbox("Scegli una lega calcio", keys_list, index=default_index)
    selected_sport_key = sport_labels[selected_sport_label]
    st.session_state.selected_sport_key = selected_sport_key
else:
    st.warning("Non sono riuscito a caricare la lista delle leghe da The Odds API. Uso Serie A di default.")
    selected_sport_key = DEFAULT_SPORT_KEY
    st.session_state.selected_sport_key = selected_sport_key

if st.button("🔎 Carica partite da The Odds API"):
    events = oddsapi_get_odds_for_sport(selected_sport_key)
    st.session_state.api_events = events

events = st.session_state.api_events

# mappiamo gli eventi in un dict "id fittizio" → evento
event_map = {}
if events:
    for i, ev in enumerate(events):
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        # a volte l'API non dà un id proprio come la tua vecchia API, quindi usiamo l'indice
        event_map[f"{i} - {home} vs {away}"] = i

if event_map:
    options = list(event_map.keys())
    # proviamo a mantenere la selezione
    if st.session_state.selected_event_id is not None:
        try:
            default_idx = options.index(st.session_state.selected_event_id)
        except ValueError:
            default_idx = 0
    else:
        default_idx = 0
    selected_event_label = st.selectbox("Partite trovate (API)", options, index=default_idx)
    st.session_state.selected_event_id = selected_event_label
else:
    selected_event_label = None

# ============================================================
# 1. INPUT MANUALE (CUORE DEL MODELLO)
# ============================================================

match_name_default = ""
parsed_from_api = {}
if selected_event_label and events:
    ev_idx = event_map[selected_event_label]
    ev = events[ev_idx]
    parsed_from_api = aggregate_odds_from_event(ev)
    match_name_default = f"{parsed_from_api.get('home','')} vs {parsed_from_api.get('away','')}"

st.subheader("1. Inserisci la partita (manuale, sempre disponibile)")
match_name = st.text_input("Nome partita", value=match_name_default)

st.subheader("1.a Linee di apertura (manuali)")
col_ap1, col_ap2 = st.columns(2)
with col_ap1:
    spread_ap = st.number_input("Spread apertura", value=0.0, step=0.25)
with col_ap2:
    total_ap = st.number_input("Total apertura", value=2.5, step=0.25)

st.subheader("1.b Linee correnti e quote (manuali, con precompilazione da API se trovata)")
col_co1, col_co2, col_co3 = st.columns(3)
with col_co1:
    spread_co = st.number_input("Spread corrente", value=0.0, step=0.25)
    odds_1 = st.number_input(
        "Quota 1",
        value=parsed_from_api["odds_1"] if parsed_from_api.get("odds_1") else 1.80,
        step=0.01
    )
with col_co2:
    total_co = st.number_input("Total corrente", value=2.5, step=0.25)
    odds_x = st.number_input(
        "Quota X",
        value=parsed_from_api["odds_x"] if parsed_from_api.get("odds_x") else 3.50,
        step=0.01
    )
with col_co3:
    odds_2 = st.number_input(
        "Quota 2",
        value=parsed_from_api["odds_2"] if parsed_from_api.get("odds_2") else 4.50,
        step=0.01
    )
    odds_btts = st.number_input(
        "Quota GG (BTTS sì)",
        value=parsed_from_api["odds_btts"] if parsed_from_api.get("odds_btts") else 1.95,
        step=0.01
    )

st.subheader("1.c Quote Over/Under (opzionali, ma le mostriamo)")
col_ou1, col_ou2 = st.columns(2)
with col_ou1:
    odds_over25 = st.number_input(
        "Quota Over 2.5 (opzionale)",
        value=parsed_from_api["odds_over25"] if parsed_from_api.get("odds_over25") else 0.0,
        step=0.01
    )
with col_ou2:
    odds_under25 = st.number_input(
        "Quota Under 2.5 (opzionale)",
        value=parsed_from_api["odds_under25"] if parsed_from_api.get("odds_under25") else 0.0,
        step=0.01
    )

st.subheader("2. xG avanzati (opzionali)")
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
    st.info("Modalità: BASE (spread/total/quote). Se inserisci xG e xGA passa in modalità AVANZATA.")
else:
    st.success("Modalità: AVANZATA (spread + total + quote + xG/xGA).")

# ============================================================
#                    CALCOLO MODELLO (SINGOLA)
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

    delta_spread = spread_co - spread_ap
    delta_total = total_co - total_ap

    st.subheader("🔁 Movimento di mercato")
    if abs(delta_spread) < 0.01 and abs(delta_total) < 0.01:
        st.write("Linee stabili: nessun movimento importante.")
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

    # PAPIRO DETTAGLIATO
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

    # archivio
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "match": match_name,
        "match_date": date.today().isoformat(),
        "league": st.session_state.selected_sport_key,
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
        "confidence": confidence,
        "esito_modello": max(
            [("1", ris_co["p_home"]), ("X", ris_co["p_draw"]), ("2", ris_co["p_away"])],
            key=lambda x: x[1]
        )[0],
        "esito_reale": "",
        "risultato_reale": "",
        "match_ok": "",
        "value_markets": "; ".join(value_markets) if value_markets else ""
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
#           ARCHIVIO MOSTRA + CANCELLAZIONE RIGA
# ============================================================
st.subheader("📁 Archivio storico analisi")
if os.path.exists(ARCHIVE_FILE):
    df_arch = pd.read_csv(ARCHIVE_FILE)
    st.dataframe(df_arch.tail(50))

    st.markdown("### 🗑️ Cancella riga dallo storico")
    if len(df_arch) > 0:
        idx_to_del = st.number_input("Indice riga da cancellare (guarda la tabella sopra)", min_value=0, max_value=len(df_arch)-1, value=len(df_arch)-1, step=1)
        if st.button("Cancella questa riga dallo storico"):
            df_arch = df_arch.drop(index=idx_to_del).reset_index(drop=True)
            df_arch.to_csv(ARCHIVE_FILE, index=False)
            st.success("Riga cancellata. Ricarica la pagina per vedere l'effetto.")
else:
    st.info("Nessun archivio trovato.")

# ============================================================
#           SCANNER GIORNALIERO (BASATO SU THE ODDS API)
# ============================================================

st.subheader("🛰️ Scanner giornaliero (The Odds API)")

if st.button("Scansiona le partite del campionato selezionato"):
    scan_events = oddsapi_get_odds_for_sport(st.session_state.selected_sport_key)
    rows_scan = []
    for ev in scan_events:
        parsed = aggregate_odds_from_event(ev)
        home = parsed.get("home", "Home")
        away = parsed.get("away", "Away")
        o1 = parsed.get("odds_1")
        ox = parsed.get("odds_x")
        o2 = parsed.get("odds_2")
        gg = parsed.get("odds_btts")

        # se manca 1X2, segna come incompleto
        if not (o1 and o2):
            rows_scan.append({
                "Partita": f"{home} vs {away}",
                "1%": "",
                "X%": "",
                "2%": "",
                "BTTS%": "",
                "Over2.5%": "",
                "Note": "1X2 incompleto"
            })
            continue

        # calcolo rapido
        base_total = 2.5
        base_spread = 0.0
        ris_scan = risultato_completo(
            base_spread, base_total,
            o1, ox if ox else 3.4, o2,
            gg if gg else 0.0,
            None, None, None, None
        )

        rows_scan.append({
            "Partita": f"{home} vs {away}",
            "1%": round(ris_scan["p_home"]*100, 1),
            "X%": round(ris_scan["p_draw"]*100, 1),
            "2%": round(ris_scan["p_away"]*100, 1),
            "BTTS%": round(ris_scan["btts"]*100, 1),
            "Over2.5%": round(ris_scan["over_25"]*100, 1),
            "Note": ", ".join(parsed.get("bookmakers_used", []))
        })

    if rows_scan:
        st.dataframe(pd.DataFrame(rows_scan))
    else:
        st.info("Nessuna partita trovata per questa lega.")

# ============================================================
#     ANALISI GIORNALIERA AUTOMATICA (SOFT)
# ============================================================

st.subheader("🤖 Analisi automatica (soft) sulla lega selezionata")

if st.button("Esegui analisi automatica"):
    auto_events = oddsapi_get_odds_for_sport(st.session_state.selected_sport_key)
    rows_auto = []
    for ev in auto_events:
        parsed = aggregate_odds_from_event(ev)
        home = parsed.get("home", "Home")
        away = parsed.get("away", "Away")
        o1 = parsed.get("odds_1") or 2.0
        ox = parsed.get("odds_x") or 3.4
        o2 = parsed.get("odds_2") or 3.5
        gg = parsed.get("odds_btts") or 0.0

        ris_auto = risultato_completo(
            0.0, 2.5,
            o1, ox, o2,
            gg,
            None, None, None, None
        )

        conf = compute_confidence(
            ris_auto,
            0.0, 0.0,
            2.5, 2.5,
            False,
            gg
        )

        rows_auto.append({
            "Partita": f"{home} vs {away}",
            "1%": round(ris_auto["p_home"]*100, 1),
            "X%": round(ris_auto["p_draw"]*100, 1),
            "2%": round(ris_auto["p_away"]*100, 1),
            "BTTS%": round(ris_auto["btts"]*100, 1),
            "Over2.5%": round(ris_auto["over_25"]*100, 1),
            "Confidence": conf,
            "Book presi": ", ".join(parsed.get("bookmakers_used", []))
        })

    if rows_auto:
        df_auto = pd.DataFrame(rows_auto)
        df_auto = df_auto.sort_values(by=["Confidence", "Over2.5%"], ascending=[False, False])
        st.dataframe(df_auto)
    else:
        st.info("Nessuna partita trovata.")