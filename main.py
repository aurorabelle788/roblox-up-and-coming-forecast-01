#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json:
for each of the chart's top 10 places, a probability for each game named being
at that place, plus the top-10 guess in order. Standard library only, no
network, no model calls, fully deterministic (same input -> same bytes).

Model (fitted on the public archive's 45 reads, 2026-10-30 to 2026-11-10;
validated on four 72-hour backtest windows, mean log score -36.98 against the
house persistence baseline's -49.19, and -33.20 against -43.51 on the official
practice window):

  Each game's rank is treated as a random walk over the scored window. Its
  mean at horizon t is its current rank plus a drift; its spread grows with
  sqrt(t). The drift has two parts, both measured from the archive's rank
  transitions:
    * a rank-band drift: top games slip a little, mid-chart games slip more,
      tail games climb (bands 1-5, 6-10, 11-20, 21-40, 41+);
    * a growth drift: a game's recent log player-count slope (24 h OLS fit),
      centred on the chart median, times -8.6 ranks per log/day. Games whose
      players are growing faster than the chart's hold their rank better;
      games growing slower fall faster. Rank trends themselves do not
      persist, so no rank-trend term is used.
  The spread is band-dependent too: rank 1-5 games move ~3.3 ranks/day,
  rank 41+ games ~19 ranks/day (all scaled by 0.75). Each place's
  probabilities are the time-average over the window of the normal rank
  marginals, scaled to sum at most 0.9 so ~10% of every place stays unnamed
  for games the history cannot name (brand-new entrants: ~8% of the practice
  window's true top-10 slots were games not on the closing chart at all).

What was tried and dropped (measured on the same backtests): playing-level
rank blends, rank-trend drift, Ornstein-Uhlenbeck mean reversion, per-game
volatility, and an age-based chart-exit hazard all scored worse or no better.
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Rank bands: (lo, hi, drift ranks/day, sigma ranks/day), measured from the
# archive's ~24 h rank transitions (3,200 pairs). Top games are sticky and
# slip slowly; tail games churn hard and climb on average.
BANDS = (
    (1, 5, 1.75, 3.33),
    (6, 10, 3.48, 6.95),
    (11, 20, 5.08, 10.60),
    (21, 40, 4.89, 15.07),
    (41, 10 ** 9, -3.18, 19.28),
)
# Fitted slope of next-day rank change on relative log player growth
# (-6.88 ranks per log/day, stable at -6.39 on early data alone), used at 1.25x.
GROWTH_BETA = -6.88 * 1.25
GROWTH_WINDOW_HOURS = 24
SIGMA_SCALE = 0.75
NAMED_SCALE = 0.9  # fraction of each place left named; rest stays unnamed
DRIFT_CLAMP = 8.0  # ranks/day
HOUR_STEP = 6
MIN_P = 0.001  # rows below this are dropped (their mass stays unnamed)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def band_of(rank):
    for lo, hi, drift, sigma in BANDS:
        if lo <= rank <= hi:
            return drift, sigma
    return 0.0, 15.0


def load_history(path):
    snaps = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
            except ValueError:
                continue
            if not snap.get("ok", True):
                continue
            rows = snap.get("up_and_coming") or []
            if not rows or not snap.get("at"):
                continue
            snaps.append((parse_time(snap["at"]), rows))
    snaps.sort(key=lambda s: s[0])
    return snaps


def log_growth(points, t_end):
    """OLS slope of log(playing) vs time (log/day) over the recent window."""
    pts = [
        (t, math.log(max(p, 1)))
        for (t, _rank, p) in points
        if (t_end - t).total_seconds() <= GROWTH_WINDOW_HOURS * 3600 and t <= t_end
    ]
    if len(pts) < 2:
        return None
    xs = [(t - t_end).total_seconds() / 86400.0 for t, _ in pts]
    ys = [v for _, v in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window") or {}
    horizon_hours = 72
    try:
        start = parse_time(window["from"])
        end = parse_time(window["until"])
        horizon_hours = max(1, int((end - start).total_seconds() // 3600))
    except (KeyError, TypeError, ValueError):
        pass
    hours = list(range(0, horizon_hours + 1, HOUR_STEP))
    if hours[-1] != horizon_hours:
        hours.append(horizon_hours)

    snaps = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not snaps:
        raise SystemExit("no usable snapshots in history.ndjson")
    t_end, last_rows = snaps[-1]
    chart = sorted(last_rows, key=lambda r: r["rank"])

    # Per-game (time, rank, playing) series, one pass over the history.
    series = {}
    for t, rows in snaps:
        for r in rows:
            series.setdefault(r["universe_id"], []).append((t, r["rank"], r["playing"]))

    # Recent player-count growth per chart game, centred on the chart median.
    growth = {}
    for r in chart:
        g = log_growth(series.get(r["universe_id"], []), t_end)
        if g is not None:
            growth[r["universe_id"]] = g
    ordered = sorted(growth.values())
    median_growth = ordered[len(ordered) // 2] if ordered else 0.0

    # Time-averaged normal rank marginals, one distribution per game.
    prob = {k: {} for k in range(1, places + 1)}
    for r in chart:
        uid = r["universe_id"]
        drift_band, sigma_band = band_of(r["rank"])
        drift = drift_band
        g = growth.get(uid)
        if g is not None:
            drift += GROWTH_BETA * (g - median_growth)
        drift = max(-DRIFT_CLAMP, min(DRIFT_CLAMP, drift))
        for h in hours:
            t = h / 24.0
            mu = r["rank"] + drift * t
            mu = max(1.0, min(float(len(chart)), mu))
            sig = 1.0 if t <= 0 else max(1.0, sigma_band * SIGMA_SCALE * math.sqrt(t))
            for k in range(1, places + 1):
                p = norm_cdf((k + 0.5 - mu) / sig) - norm_cdf((k - 0.5 - mu) / sig)
                if p > 0:
                    prob[k][uid] = prob[k].get(uid, 0.0) + p / len(hours)

    probabilities = []
    for k in range(1, places + 1):
        total = sum(prob[k].values())
        for uid, p in sorted(prob[k].items(), key=lambda kv: (-kv[1], kv[0])):
            if total > 1.0:
                p /= total
            p *= NAMED_SCALE
            if p >= MIN_P:
                probabilities.append(
                    {"universe_id": uid, "place": k, "p": round(p, 4)}
                )

    # Top 10: most likely game at each place, in place order, deduplicated.
    top10 = []
    for k in range(1, places + 1):
        for uid, _p in sorted(prob[k].items(), key=lambda kv: (-kv[1], kv[0])):
            if uid not in top10:
                top10.append(uid)
                break
    for r in chart:  # fill any gap from the closing chart order
        if len(top10) >= places:
            break
        if r["universe_id"] not in top10:
            top10.append(r["universe_id"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"probabilities": probabilities, "top10": top10[:places]},
                separators=(",", ":"),
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
