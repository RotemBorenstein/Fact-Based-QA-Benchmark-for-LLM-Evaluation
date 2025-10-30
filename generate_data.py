import pandas as pd
import json
from urllib.parse import unquote, quote
import requests
from bs4 import BeautifulSoup
import unicodedata
from collections import defaultdict
import hashlib
import re
import io
import os


LEAGUE_RANK_INPUT = {
    1: "Premier_League",
    2: "La_Liga",
    3: "Bundesliga",
    4: "Serie_A",
    5: "Ligue_1",
    6: "Campeonato_Brasileiro_Série_A",
    7: "Major_League_Soccer",
    8: "Liga_MX",
    9: "Primeira_Liga",
    10: "Belgian_Pro_League",
    11: "EFL_Championship",
    12: "Argentine_Primera_División",
    13: "Eredivisie",
    14: "Danish_Superliga",
    15: "Saudi_Pro_League",
    16: "J1_League",
    17: "Chinese_Super_League",
    18: "Russian_Premier_League",
    19: "Süper_Lig",
    20: "Chilean_Primera_División",
    21: "A-League_Men",
    22: "K_League_1",
    23: "Categoría_Primera_A",
    24: "Persian_Gulf_Pro_League",
    25: "Premier_Soccer_League_(South_Africa)",
    26: "Egyptian_Premier_League",
    27: "UAE_Pro_League",
    28: "Scottish_Premiership",
    29: "Super_League_Greece",
    30: "Qatar_Stars_League",
    31: "Ukrainian_Premier_League",
    32: "Austrian_Football_Bundesliga",
    33: "Swiss_Super_League",
    34: "Ekstraklasa",
    35: "Czech_First_League",
    36: "Liga_I",
    37: "Croatian_Football_League",
    38: "Serbian_SuperLiga",
    39: "Eliteserien",
    40: "Allsvenskan",
    41: "Veikkausliiga",
    42: "Scottish_Championship",
    43: "Indian_Super_League",
    44: "MLS_Next_Pro",
    45: "Canadian_Premier_League",
    46: "Campeonato_Brasileiro_Série_B",
    47: "Liga_1_(Indonesia)",
    48: "Thai_League_1",
    49: "V.League_1",
    50: "Malaysia_Super_League"
}

# handle duplicates
LEAGUE_BEST_RANK = {}
for k in sorted(LEAGUE_RANK_INPUT.keys()):
    name = LEAGUE_RANK_INPUT[k]
    if name not in LEAGUE_BEST_RANK:
        LEAGUE_BEST_RANK[name] = k

def _league_key(s: str) -> str:
    """underscore style"""
    return s.strip().replace(" ", "_").replace("—", "-")

def popularity_tier(league_name: str) -> int:
    """
    Return 0 for ranks 1–10, 1 for 11–50, 2 for everything else.
    Input can be 'Premier_League' or 'Premier League'.
    """
    key = _league_key(league_name)
    rank = LEAGUE_BEST_RANK.get(key)
    if rank is None:
        return 2
    return 0 if rank <= 10 else 1

def popularity_score(league_name: str) -> int:
    """ use popularity_tier as the score component (0/1/2)"""
    return popularity_tier(league_name)


NOW_YEAR = 2025

def end_year_from_season(season: str) -> int:
    """
    '2015–16' -> 2016, '2019-2020' -> 2020, '2023' -> 2023.
    Handles hyphen/en-dash variations and 2-digit tails.
    """
    if not season:
        return NOW_YEAR
    s = season.replace('—', '–').replace('-', '–')
    m = re.match(r'(\d{4})\s*–\s*(\d{2,4})', s)
    if m:
        y1 = int(m.group(1))
        tail = m.group(2)
        y2 = int(tail) if len(tail) == 4 else (y1 // 100) * 100 + int(tail)
        if y2 < y1:
            y2 += 100  # rare century wrap
        return y2
    m = re.search(r'(\d{4})', s)
    return int(m.group(1)) if m else NOW_YEAR

def recency_score_single(season: str) -> int:
    age = max(0, NOW_YEAR - end_year_from_season(season))
    if age <= 5:
        return 0
    if age <= 10:
        return 1
    return 2

def recency_score_two(season1: str, season2: str) -> int:
    """take the max of the two single-season scores"""
    return max(recency_score_single(season1), recency_score_single(season2))



def _pair_closeness_score(p1: int, p2: int) -> int:
    gap = abs(int(p1) - int(p2))
    if gap <= 1:
        return 1   # adjacent = hardest
    if gap <= 3:
        return 0   # close = medium
    return 0       # far apart = easy
def _bucket(total: int) -> str:
    """0–1: easy, 2–3: medium, >=4: hard."""
    if total <= 2:
        return "easy"
    if total <= 4:
        return "medium"
    return "hard"

def compute_difficulty_single(league: str, season: str, question_type: str, is_split: bool=False, stat: str=None) -> str:
    stat_score = 0
    if question_type == "single_season_stat" and stat:
        stat_score = STAT_DIFFICULTY_SCORE.get(stat, 1)
    total = (
        TEMPLATE_SCORE[question_type] +
        popularity_score(league) +
        recency_score_single(season) +
        (1 if is_split else 0) +
        stat_score
    )
    return _bucket(total)

def compute_difficulty_two(league: str, season1: str, season2: str, is_split: bool=False) -> str:
    total = (
        TEMPLATE_SCORE["two_seasons_placement"] +
        popularity_score(league) +
        recency_score_two(season1, season2) +
        (1 if is_split else 0)
    )
    return _bucket(total)


def compute_difficulty_pairwise(league: str, season: str, is_split: bool, p1: int, p2: int) -> str:
    total = (
        TEMPLATE_SCORE["pairwise_higher"] +
        popularity_score(league) +
        recency_score_single(season) +
        (1 if is_split else 0) +
        _pair_closeness_score(p1, p2)
    )
    return _bucket(total)


def normalize_team_name(name: str) -> str:
    name = clean_team_name(str(name))
    name = ''.join(
        ch for ch in unicodedata.normalize('NFKD', name)
        if not unicodedata.combining(ch) and ch.isalnum()
    ).lower()
    return name

def season_phrase(league: str, season: str, is_split: bool) -> str:
    return f"the regular season of {league} {season}" if is_split else f"{league} {season}"

def get_stat_col(df, short_key):
    """Find the matching column name in the DataFrame for a given stat short_key."""
    for cand in stat_column_map[short_key]:
        for col in df.columns:
            if str(col).strip().lower() == cand:
                return col
    return None
def select_positions(team_count, n):
    return [i for i in range(n, team_count+1, n)]
def detect_split_league(url):
    playoff_keywords = [
        "post-season", "upper group", "lower group", "regular season", "regular season table"
    ]
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, 'html.parser')

    # Search all headings (h1-h4) and table captions
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'caption']):
        if tag and tag.text:
            text = tag.text.lower()
            if any(keyword in text for keyword in playoff_keywords):
                return True
    return False


def encode_wikipedia_url(url):
    if "wiki/" in url:
        prefix, title = url.split("wiki/", 1)
        title_encoded = quote(title)
        return prefix + "wiki/" + title_encoded
    return url
def fetch_teams_from_wikipedia(url):
    try:
        # Add User-Agent to avoid 403
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text

        tables = pd.read_html(io.StringIO(html), header=0)
    except Exception as e:
        return None, None, f"Error reading HTML tables: {e}"

    for table in tables:
        # Flatten multiindex columns if needed, use the first element of tuple if column is a tuple
        columns_flat = [col[0] if isinstance(col, tuple) else col for col in table.columns]
        table.columns = columns_flat

        pos_candidates = [col for col in table.columns if str(col).strip().lower() in ['pos', 'position', 'no.']]
        team_candidates = [col for col in table.columns if str(col).strip().lower() in ['team', 'club', 'teamvte']]

        if pos_candidates and team_candidates:
            pos_col = pos_candidates[0]
            team_col = team_candidates[0]
            result = []
            for _, row in table.iterrows():
                pos = str(row[pos_col]).strip()
                team = str(row[team_col]).strip()
                # skip empty and summary rows
                if pos.isdigit() and team and team.lower() not in ["", "total"]:
                    result.append({"position": pos, "team": team})
            if result and len(result) > 4:  # at least a reasonable number of teams in a league
                return result, table, None

    return None, None, "No suitable table found"




def parse_league_and_season(url):
    title = unquote(url.split('/')[-1]).replace('_', ' ')
    m = re.match(r'(?P<season>\d{4}(?:–\d{2,4})?) (?P<league>.+)', title)
    if m:
        return m.group('league').strip(), m.group('season').strip()
    else:
        return title.strip(), ""

def clean_team_name(team):
    return re.sub(r'\s*\([^)]*\)\s*$', '', team).strip()

def create_prompt(league_name, season, position):
    return f"What team finished in place {position} in {league_name} {season}?"



league_season_cache = defaultdict(dict)  # {league: {season: {"teams": [...], "is_split": bool}}}
urls = []
n = 3
n_stats = 2
mixed_seasons_n = 4
n_pairwise_comparison = 5
GAPS = [1, 2, 3]
TEMPLATE_SCORE = {
    "single_season_placement": 0,
    "single_season_stat": 1,
    "two_seasons_placement": 1,
    "pairwise_higher": 0
}
stat_columns = ['GD', 'W', 'D', 'L', 'GF', 'Pts', 'GA']
stat_column_map = {
    'GD':  ['gd', 'goal difference'],
    'W':   ['w', 'wins'],
    'D':   ['d', 'draws'],
    'L':   ['l', 'losses'],
    'GF':  ['gf', 'goals for'],
    'GA':  ['ga', 'goals against'],
    'Pts': ['pts', 'points'],
}

stat_prompt_map = {
    'W': 'number of wins',
    'D': 'number of draws',
    'L': 'number of losses',
    'GF': 'number of goals scored by',
    'GA': 'number of goals conceded by',
    'GD': 'goal difference',
    'Pts': 'number of points',
}

STAT_DIFFICULTY_SCORE = {
    "Pts": 1,
    "W": 1,
    "GF": 2,
    "GA": 2,
    "L": 2,
    "D": 2,
    "GD": 2,
}


with open('leagues.txt', 'r', encoding='utf-8') as f:
    urls = [line.strip() for line in f if line.strip()]

dataset = []
errors = []

for url in urls:
    teams, df, error = fetch_teams_from_wikipedia(encode_wikipedia_url(url))
    if error:
        errors.append({"url": url, "error": error})
        continue

    league_name, season = parse_league_and_season(url)
    is_split_league = detect_split_league(url)
    team_count = len(teams)
    positions_to_sample = select_positions(team_count, n)
    positions_to_sample_stats = select_positions(team_count, n_stats)

    league_season_cache[league_name][season] = {
        "teams": teams,
        "is_split": is_split_league
    }

    for team_info in teams:
        try:
            position_int = int(team_info['position'])
        except Exception:
            continue  # skip any rows where position is not an integer
        if position_int in positions_to_sample:
            if is_split_league:
                prompt = (
                    f"What team finished in place {team_info['position']} in the regular season of {league_name} {season}? "
                    "Your answer should include only the team name. Use the most common name, abbreviation, or nickname for the team."
                )
            else:
                prompt = (
                    f"What team finished in place {team_info['position']} in {league_name} {season}? "
                    "Your answer should include only the team name. Use the most common name, abbreviation, or nickname for the team."
                )
            prompt = prompt.replace(' ?', '?')
            answer = clean_team_name(team_info['team'])
            question_type = "single_season_placement"
            dataset.append({
                "prompt": prompt,
                "answer": clean_team_name(team_info["team"]),
                "league": league_name,
                "season": season,
                "position": str(team_info["position"]),
                "question_type": question_type,
                "difficulty": compute_difficulty_single(
                    league=league_name,
                    season=season,
                    question_type=question_type,
                    is_split=is_split_league
                ),
            })

    if df is not None:
        # Pair positions_to_sample_stats with stat_columns, cycle stats if more positions than stats
        paired_stats = list(
            zip(
                positions_to_sample_stats,
                (stat_columns * ((len(positions_to_sample_stats) // len(stat_columns)) + 1))[
                :len(positions_to_sample_stats)]
            )
        )
        for position, stat in paired_stats:
            stat_col = get_stat_col(df, stat)
            if not stat_col:
                continue
            # Find the matching team info for the position
            team_info = next((t for t in teams if int(t['position']) == position), None)
            if not team_info:
                continue
            stat_val = df[df[df.columns[0]].astype(str).str.strip() == str(position)][stat_col].values
            if stat_val.size > 0:
                prompt_stat = stat_prompt_map.get(stat, stat)

                if is_split_league:
                    if prompt_stat.startswith('number of goals scored by') or prompt_stat.startswith(
                            'number of goals conceded by') or prompt_stat.startswith('number of games played'):
                        prompt = (
                            f"What is the {prompt_stat} the team that finished in place {position} in the regular season of {league_name} {season}? "
                            "Your answer should include only the number."
                        )
                    else:
                        prompt = (
                            f"What is the {prompt_stat} of the team that finished in place {position} in the regular season of {league_name} {season}? "
                            "Your answer should include only the number."
                        )
                else:
                    if prompt_stat.startswith('number of goals scored by') or prompt_stat.startswith(
                            'number of goals conceded by') or prompt_stat.startswith('number of games played'):
                        prompt = (
                            f"What is the {prompt_stat} the team that finished in place {position} in {league_name} {season}? "
                            "Your answer should include only the number."
                        )
                    else:
                        prompt = (
                            f"What is the {prompt_stat} of the team that finished in place {position} in {league_name} {season}? "
                            "Your answer should include only the number."
                        )
                prompt = prompt.replace(' ?', '?')

                question_type = "single_season_stat"
                dataset.append({
                    "prompt": prompt,
                    "answer": str(stat_val[0]),
                    "league": league_name,
                    "season": season,
                    "position": str(position),
                    "stat": stat_col,
                    "question_type": question_type,
                    "difficulty": compute_difficulty_single(
                        league=league_name,
                        season=season,
                        question_type=question_type,
                        is_split=is_split_league,
                        stat=stat
                    ),
                })


    # Compare adjacent positions

    # Build position -> team map
    pos_to_team = {}
    for t in teams:
        pos_str = str(t.get("position", "")).strip()
        if pos_str.isdigit():
            pos_to_team[int(pos_str)] = clean_team_name(t.get("team", ""))

    pair_starts = select_positions(team_count, n_pairwise_comparison)

    for idx, p1 in enumerate(pair_starts):
        chosen_idx = idx % len(GAPS)
        chosen_gap = GAPS[chosen_idx]

        # Try chosen_gap first, then fall back to the next gaps in cycle if it doesn't fit
        gap = None
        for off in range(len(GAPS)):
            g_try = GAPS[(chosen_idx + off) % len(GAPS)]
            if p1 + g_try <= team_count:
                gap = g_try
                break

        if gap is None:
            continue

        p2 = p1 + gap
        if p1 not in pos_to_team or p2 not in pos_to_team:
            continue

        team1 = pos_to_team[p1]
        team2 = pos_to_team[p2]
        gold = team1 if p1 < p2 else team2

        phrase = season_phrase(league_name, season, is_split_league)
        prompt = (
            f"Which team finished higher in {phrase}: {team1} or {team2}? "
            "Your answer should include only the team name. "
            "Use the most common name, abbreviation, or nickname for the team."
        ).replace(' ?', '?')

        dataset.append({
            "prompt": prompt,
            "answer": gold,
            "league": league_name,
            "season": season,
            "team1": team1, "position1": str(p1),
            "team2": team2, "position2": str(p2),
            "question_type": "pairwise_higher",
            "difficulty": compute_difficulty_pairwise(
                league=league_name,
                season=season,
                is_split=is_split_league,
                p1=p1, p2=p2
            ),
        })

# cross-season prompts
sample_from = 'first'
seen_pairs = set()
for league_name, seasons_dict in league_season_cache.items():
    seasons = sorted(seasons_dict.keys())  # deterministic order
    for i in range(len(seasons)):
        for j in range(i + 1, len(seasons)):
            s1, s2 = seasons[i], seasons[j]
            d1, d2 = seasons_dict[s1], seasons_dict[s2]
            teams1, teams2 = d1["teams"], d2["teams"]

            if not teams1 or not teams2:
                continue

            # Build lookups for each season: normalized name -> (orig_name, position)
            lookup1 = {}
            for t in teams1:
                if str(t.get("position", "")).isdigit():
                    lookup1[normalize_team_name(t["team"])] = (t["team"], int(t["position"]))
            lookup2 = {}
            for t in teams2:
                if str(t.get("position", "")).isdigit():
                    lookup2[normalize_team_name(t["team"])] = (t["team"], int(t["position"]))


            if sample_from == 'first':
                sample_source = (s1, d1, teams1, lookup1, lookup2)
                sample_target = (s2, d2)
            else:
                sample_source = (s2, d2, teams2, lookup2, lookup1)
                sample_target = (s1, d1)

            src_season, src_meta, src_teams, src_lookup, tgt_lookup = sample_source
            tgt_season, tgt_meta = sample_target


            pos_sample = select_positions(len(src_teams), mixed_seasons_n)

            for p in pos_sample:
                src_team_rec = next((t for t in src_teams
                                     if str(t.get("position", "")).isdigit() and int(t["position"]) == p), None)
                if not src_team_rec:
                    continue

                team_orig = src_team_rec["team"]
                team_norm = normalize_team_name(team_orig)


                if team_norm not in tgt_lookup:
                    continue

                _, target_pos = tgt_lookup[team_norm]

                phrase_src = season_phrase(league_name, src_season, src_meta["is_split"])
                phrase_tgt = season_phrase(league_name, tgt_season, tgt_meta["is_split"])

                key = (league_name, team_norm, src_season, p, tgt_season, target_pos)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                prompt = (
                    f"What team finished in both place {p} in {phrase_src}, "
                    f"and in place {target_pos} in {phrase_tgt}?"
                    "Your answer should include only the team name. "
                    "Use the most common name, abbreviation, or nickname for the team."
                ).replace(' ?', '?')


                question_type = "two_seasons_placement"
                is_split_season1 = src_meta["is_split"]
                is_split_season2 = tgt_meta["is_split"]

                dataset.append({
                    "prompt": prompt,
                    "answer": clean_team_name(team_orig),
                    "league": league_name,
                    "season1": src_season, "position1": str(p),
                    "season2": tgt_season, "position2": str(target_pos),
                    "question_type": question_type,
                    "difficulty": compute_difficulty_two(
                        league=league_name,
                        season1=src_season,
                        season2=tgt_season,
                        is_split=(is_split_season1 or is_split_season2)  # or False if N/A
                    ),
                })




def _to_eval_format(ex):
    ex = dict(ex)
    ex.pop("candidates", None)

    if "answer" in ex:
        ex["gold_answer"] = str(ex.pop("answer"))

    prompt = ex.get("prompt", "")
    if "Your answer should include only the number" in prompt:
        ex["answer_type"] = "number"
    else:
        ex["answer_type"] = "team"

    if "id" not in ex:
        key = f"{prompt}||{ex.get('gold_answer','')}"
        ex["id"] = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

    return ex

dataset_eval = [_to_eval_format(e) for e in dataset]
os.makedirs("data", exist_ok=True)
with open("data/examples.jsonl", "w", encoding="utf8") as f:
    for example in dataset_eval:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

if errors:
    with open("errors.log", "w", encoding="utf8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")

