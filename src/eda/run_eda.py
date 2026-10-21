"""Предметный разведочный анализ исходной части (data/interim/initial.csv).

Результаты: reports/eda_summary.json (числа, на которые ссылается отчёт)
и рисунки в reports/figures. Анализ выполняется после удаления полных
дубликатов, как в конвейере.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import stats

T = "booking_complete"
FIG = Path("reports/figures")
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10})
CM = 1 / 2.54


def save(fig, path):
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    im = ImageOps.expand(im, border=12, fill="white")
    ImageOps.expand(im, border=3, fill="black").save(path)


def mantel_haenszel(df, strata, flag, min_n=20):
    """Отношение шансов Мантеля — Хензеля для бинарного флага внутри страт."""
    num = den = 0.0
    used = n = 0
    sa = se = sv = 0.0
    wa = wo = na = no = 0.0
    for _, g in df.groupby(strata):
        a_ = g[g[flag]]
        o_ = g[~g[flag]]
        if len(a_) < min_n or len(o_) < min_n:
            continue
        a, b = a_[T].sum(), len(a_) - a_[T].sum()
        c, d = o_[T].sum(), len(o_) - o_[T].sum()
        num += a * d / len(g)
        den += b * c / len(g)
        N = len(g)
        m1, n1 = a + c, a + b
        sa += a
        se += n1 * m1 / N
        sv += n1 * (c + d) * m1 * (b + d) / (N**2 * (N - 1))
        wa += a
        na += a + b
        wo += c
        no += c + d
        used += 1
        n += len(g)
    chi = (abs(sa - se) - 0.5) ** 2 / sv
    return {"or_mh": round(num / den, 3), "strata": used, "rows": int(n),
            "cmh_p": float(stats.chi2.sf(chi, 1)),
            "conv_flag_in_strata": round(wa / na, 4), "conv_rest_in_strata": round(wo / no, 4)}


def chi2(a, b):
    ct = pd.crosstab(a, b)
    s, p, *_ = stats.chi2_contingency(ct)
    return float(s), float(p)


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    s = stats.chi2_contingency(ct)[0]
    return float(np.sqrt(s / (ct.values.sum() * (min(ct.shape) - 1))))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv("data/interim/initial.csv")
    feats = [c for c in raw.columns if c not in ("booking_id", T)]
    df = raw.drop_duplicates(subset=feats + [T]).copy()
    S = {"rows_initial": len(raw), "rows_dedup": len(df), "positives": int(df[T].sum()),
         "positive_share": round(df[T].mean(), 4)}
    base = df[T].mean()

    num = ["num_passengers", "purchase_lead", "length_of_stay", "flight_hour", "flight_duration"]
    S["numeric"] = {}
    for c in num:
        s = df[c]
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
        S["numeric"][c] = {"min": float(s.min()), "median": float(s.median()), "max": float(s.max()),
                           "iqr_out_share": round(float(out), 4), "skew": round(float(s.skew()), 2),
                           "median_pos": float(s[df[T] == 1].median()),
                           "median_neg": float(s[df[T] == 0].median()),
                           "mw_p": float(stats.mannwhitneyu(s[df[T] == 1], s[df[T] == 0]).pvalue)}
    S["stay_7_16"] = int(df.length_of_stay.between(7, 16).sum())
    S["lead_over_365"] = int((df.purchase_lead > 365).sum())
    S["oneway_with_stay"] = int(((df.trip_type == "OneWay") & (df.length_of_stay > 0)).sum())
    S["not_set"] = int((df.booking_origin == "(not set)").sum())
    S["routes"] = int(df.route.nunique())
    S["routes_rare"] = int((df.route.value_counts() < 10).sum())
    S["countries"] = int(df.booking_origin.nunique())
    S["countries_rare"] = int((df.booking_origin.value_counts() < 10).sum())
    S["duration_per_route_max"] = int(df.groupby("route").flight_duration.nunique().max())

    # --- гипотезы ---
    df["is_au"] = df.booking_origin.eq("Australia")
    df["mobile"] = df.sales_channel.eq("Mobile")
    df["short_stay"] = df.length_of_stay.le(6)
    df["late"] = df.purchase_lead.le(7)
    df["n_extras"] = df[["wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"]].sum(axis=1)
    H = {}
    H["H1_country_not_route"] = {"conv_au": round(df[df.is_au][T].mean(), 4),
                                 "conv_other": round(df[~df.is_au][T].mean(), 4),
                                 **mantel_haenszel(df, "route", "is_au")}
    ct = pd.crosstab(df.n_extras, df[T])
    x, n, y = ct.index.values, ct.sum(axis=1).values, ct[1].values
    p = y.sum() / n.sum()
    z = (x * (y - n * p)).sum() / np.sqrt(p * (1 - p) * ((n * x**2).sum() - (n * x).sum() ** 2 / n.sum()))
    H["H2_extras_trend"] = {"conv_by_n": {int(k): round(v, 4) for k, v in df.groupby("n_extras")[T].mean().items()},
                            "z": round(float(z), 2), "p": float(2 * stats.norm.sf(abs(z)))}
    H["H3_short_stay"] = {"conv_short": round(df[df.short_stay][T].mean(), 4),
                          "conv_long": round(df[~df.short_stay][T].mean(), 4),
                          "chi2_p": chi2(df.short_stay, df[T])[1],
                          **mantel_haenszel(df, "booking_origin", "short_stay")}
    H["H4_mobile"] = {"conv_mobile": round(df[df.mobile][T].mean(), 4),
                      "conv_internet": round(df[~df.mobile][T].mean(), 4),
                      "chi2_p": chi2(df.mobile, df[T])[1],
                      **mantel_haenszel(df, "booking_origin", "mobile")}
    H["H5_last_minute"] = {"conv_late": round(df[df.late][T].mean(), 4),
                           "conv_early": round(df[~df.late][T].mean(), 4),
                           "chi2_p": chi2(df.late, df[T])[1],
                           **mantel_haenszel(df, "booking_origin", "late")}
    H["day_of_week_p"] = chi2(df.flight_day, df[T])[1]
    S["hypotheses"] = H

    # --- взаимосвязи ---
    S["spearman"] = df[num].corr(method="spearman").round(2).to_dict()
    cats = ["sales_channel", "trip_type", "flight_day", "route", "booking_origin", T]
    S["cramers_v"] = {a: {b: round(cramers_v(df[a], df[b]), 2) for b in cats} for a in cats}

    # --- рисунок: распределения ---
    fig, ax = plt.subplots(1, 2, figsize=(15.5 * CM, 6.0 * CM))
    bins = np.arange(0, 400, 15)
    for cls, st, lab in ((0, "step", "не выписана"), (1, "stepfilled", "выписана")):
        kw = dict(color="0.25", lw=1.2) if cls == 0 else dict(color="0.75")
        ax[0].hist(df[df[T] == cls].purchase_lead.clip(upper=399), bins=bins, density=True,
                   histtype=st, label=lab, **kw)
    ax[0].set_xlabel("purchase_lead, дней")
    ax[0].set_ylabel("Плотность")
    ax[0].legend(frameon=False, fontsize=9)
    vc = df.length_of_stay.clip(upper=40).value_counts().sort_index().reindex(range(41), fill_value=0)
    ax[1].bar(vc.index, vc.values, color="0.55", width=0.8)
    ax[1].axvspan(6.5, 16.5, color="0.9", zorder=0)
    ax[1].text(11.5, vc.max() * 0.8, "нет значений\n7–16", ha="center", fontsize=9)
    ax[1].set_xlabel("length_of_stay, дней (> 40 — в крайнем столбце)")
    ax[1].set_ylabel("Число броней")
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ",")))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.5)
    save(fig, FIG / "eda_distributions.png")

    # --- рисунок: конверсия по сегментам ---
    fig, ax = plt.subplots(2, 2, figsize=(15.5 * CM, 10.5 * CM))
    top = df.booking_origin.value_counts().head(8).index
    c = df[df.booking_origin.isin(top)].groupby("booking_origin")[T].mean().reindex(top)
    names = {"Australia": "Австралия", "Malaysia": "Малайзия", "South Korea": "Юж. Корея",
             "Japan": "Япония", "China": "Китай", "Indonesia": "Индонезия",
             "Taiwan": "Тайвань", "Thailand": "Таиланд", "India": "Индия",
             "New Zealand": "Н. Зеландия", "Singapore": "Сингапур"}
    panels = [
        (c, [names.get(i, i) for i in c.index], "а) страна создания брони (8 крупнейших)"),
        (df.groupby(pd.cut(df.length_of_stay, [-1, 3, 6, 30, 60, 10**4]), observed=True)[T].mean(),
         ["0–3", "4–6", "17–30", "31–60", "> 60"], "б) длительность пребывания, дней"),
        (df.groupby("n_extras")[T].mean(), ["0", "1", "2", "3"], "в) число дополнительных услуг"),
        (df.groupby(pd.cut(df.purchase_lead, [-1, 7, 30, 90, 180, 365, 10**4]), observed=True)[T].mean(),
         ["0–7", "8–30", "31–90", "91–180", "181–365", "> 365"], "г) глубина покупки, дней"),
    ]
    for a, (vals, labels, title) in zip(ax.flat, panels):
        a.bar(range(len(vals)), vals.values * 100, color="0.6", edgecolor="black", lw=0.6)
        a.axhline(base * 100, color="black", ls="--", lw=0.8)
        a.set_xticks(range(len(vals)), labels, fontsize=8.5,
                     rotation=35 if len(labels) >= 6 else 0, ha="right" if len(labels) >= 6 else "center")
        a.set_title(title, fontsize=10)
        a.set_ylabel("Доля выписанных, %", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, FIG / "eda_conversion.png")

    # --- рисунок: взаимосвязи ---
    fig, ax = plt.subplots(1, 2, figsize=(15.5 * CM, 6.8 * CM))
    sp = df[num].corr(method="spearman").values
    short = ["пасс.", "lead", "stay", "час", "длит."]
    cv = np.array([[S["cramers_v"][a][b] for b in cats] for a in cats])
    cshort = ["канал", "тип", "день", "марш.", "страна", "цель"]
    for a, m, lab, ttl in ((ax[0], sp, short, "а) Спирмен, числовые"),
                           (ax[1], cv, cshort, "б) V Крамера, категориальные")):
        a.imshow(np.abs(m), cmap="Greys", vmin=0, vmax=1.6)
        for i in range(len(lab)):
            for j in range(len(lab)):
                a.text(j, i, f"{m[i, j]:.2f}".replace(".", ",").replace("-", "−"), ha="center", va="center", fontsize=8)
        a.set_xticks(range(len(lab)), lab, fontsize=8.5)
        a.set_yticks(range(len(lab)), lab, fontsize=8.5)
        a.set_title(ttl, fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "eda_relations.png")

    json.dump(S, open("reports/eda_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print(json.dumps(S["hypotheses"], ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
