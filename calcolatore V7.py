def oddsapi_get_events_for_league(league_key: str) -> List[dict]:
    """
    Prende gli eventi per una lega di calcio.
    markets: h2h, totals, spreads, both_teams_to_score → per 1X2, over/under, DNB e GG.
    """
    try:
        r = requests.get(
            f"{THE_ODDS_BASE}/sports/{league_key}/odds",
            params={
                "apiKey": THE_ODDS_API_KEY,
                "regions": "eu,uk",
                # 👇 aggiunto BTTS
                "markets": "h2h,totals,spreads,both_teams_to_score",
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