#!/usr/bin/env python3
"""사전 등록(PREREGISTRATION.md)의 게이트 1·2·3 을 n=5 로 판정한다.

    python scripts/je_loco/judge_gates.py results/full_matrix results/full_matrix_hard

판정 규칙은 데이터를 보기 전에 고정된 것이며 이 스크립트는 그것을 그대로 집행할 뿐이다.
  게이트 1  무결손 대등성  : level 0 성공률, 동등 마진 |Δ| < 2.0pp
  게이트 2  공간 결손      : occlusion AULC — 조건간 격차 > 최대 조건내 시드폭 일 때만 신호
  게이트 3  시간 결손      : freeze/latency/lowfps AULC, 사전 예측 jepa > recon 및 jepa > scratch
"""
import argparse, csv, glob, os, re, sys
from collections import defaultdict

CONDS = ["jepa", "recon", "scratch"]
MARGIN_PP = 2.0            # 게이트 1 동등 마진 (사전 등록)
SPATIAL = ["occlusion"]    # dropout 은 판정력 없음이 확인되어 사전 등록 단계에서 제외
TEMPORAL = ["freeze", "latency", "lowfps"]

RUN_RE = re.compile(r"_D_(jepa|recon|scratch)_s(\d+)_model_")
V2 = False


def use_v2():
    """v2 는 비교축이 다르다 — 조건 scratch→none, 런 이름 D_→V2_,
    게이트 1 판정도 절대 마진이 아니라 시드폭 기준(사전등록 v2 §4)."""
    global CONDS, RUN_RE, V2
    CONDS = ["jepa", "recon", "none"]
    RUN_RE = re.compile(r"_V2_(jepa|recon|none)_s(\d+)_model_")
    V2 = True


def load(root):
    """{(deg, cond, seed): [(level, success), ...]}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*_curve_*.csv"))):
        deg = os.path.basename(path).split("_curve_")[0]
        m = RUN_RE.search(os.path.basename(path))
        if not m:
            print(f"  ! 이름 파싱 실패, 건너뜀: {os.path.basename(path)}", file=sys.stderr)
            continue
        cond, seed = m.group(1), int(m.group(2))
        with open(path) as f:
            rd = csv.DictReader(f)
            lvl = rd.fieldnames[0]          # 첫 열 이름 = 결손 종류
            rows = [(float(r[lvl]), float(r["success_rate"])) for r in rd]
        out[(deg, cond, seed)] = sorted(rows)
    return out


def aulc(curve):
    """레벨 축 [0,1] 위 성공률의 사다리꼴 평균. 레벨 간격이 균일하지 않아도 맞다."""
    if len(curve) < 2:
        return float("nan")
    area = sum((curve[i + 1][0] - curve[i][0]) * (curve[i + 1][1] + curve[i][1]) / 2
               for i in range(len(curve) - 1))
    return area / (curve[-1][0] - curve[0][0])


def l50(curve):
    """성공률이 처음 50% 아래로 내려가는 레벨(선형보간). 끝까지 안 내려가면 None."""
    for i in range(len(curve) - 1):
        (x0, y0), (x1, y1) = curve[i], curve[i + 1]
        if y0 >= 0.5 > y1:
            return x0 + (y0 - 0.5) / (y0 - y1) * (x1 - x0)
    return None


def stat(vals):
    return sum(vals) / len(vals), min(vals), max(vals)


def fmt_spread(vals, scale=100, unit="pp"):
    m, lo, hi = stat(vals)
    return f"{m*scale:6.2f}  [{lo*scale:5.2f}, {hi*scale:5.2f}]  폭 {(hi-lo)*scale:4.2f}{unit}"


def gate1(data, label):
    print(f"\n{'='*78}\n게이트 1 · 무결손 대등성 ({label})  — 주 주장\n{'='*78}")
    # 결손 종류와 무관하게 level 0 은 항등이어야 한다. 그 항등성부터 확인한다.
    per = defaultdict(list)
    ident = defaultdict(set)
    for (deg, cond, seed), curve in data.items():
        if deg == "abl" or curve[0][0] != 0.0:
            continue      # abl 은 ẑ 를 절제한 곡선 — 무결손 성능이 아니다(게이트 4 전용)
        ident[(cond, seed)].add(round(curve[0][1], 4))
        per[cond].append((seed, deg, curve[0][1]))
    bad = {k: v for k, v in ident.items() if len(v) > 1}
    if bad:
        print("  ! level 0 항등성 위반 — 결손 종류에 따라 무결손 성능이 다르다:")
        for (c, s), v in sorted(bad.items()):
            print(f"      {c}_s{s}: {sorted(v)}")
    else:
        print("  level 0 항등성 확인 (모든 결손 종류에서 동일) ✓")

    means = {}
    print(f"\n  {'조건':<9} {'n':>3}  {'성공률 %':>8}  {'[min, max]':>16}  시드폭")
    for c in CONDS:
        seeds = sorted({s for s, _, _ in per[c]})
        vals = [next(v for s, _, v in per[c] if s == sd) for sd in seeds]
        means[c] = stat(vals)[0]
        print(f"  {c:<9} {len(seeds):>3}  {fmt_spread(vals)}")
    spread = max(stat([next(v for s, _, v in per[c] if s == sd)
                       for sd in sorted({s for s, _, _ in per[c]})])[2]
                 - stat([next(v for s, _, v in per[c] if s == sd)
                         for sd in sorted({s for s, _, _ in per[c]})])[1]
                 for c in CONDS) * 100

    d = (means["jepa"] - means["recon"]) * 100
    if V2:
        # v1 에서 절대 마진(2.0pp)을 미리 박았다가 실제 시드폭(2.6~7.8pp)이 그보다 커져
        # 검정력 논거가 무너졌다. v2 는 시드폭 기준으로 통일한다(자기교정).
        print(f"\n  Δ(jepa − recon) = {d:+.2f}pp   최대 조건내 시드폭 {spread:.2f}pp")
        if d >= 0:
            print("  판정: ★ 통과 — jepa 가 recon 이상 (특권 정보 없이 대등 이상)")
        elif abs(d) < spread:
            print("  판정: ★ 통과 — 격차가 시드폭 이내로 동등")
        else:
            print(f"  판정: ✗ 실패 — jepa 가 recon 에 {abs(d):.2f}pp 열세 (시드폭 초과)")
        print(f"  전제 확인: {CONDS[2]} 평균 {means[CONDS[2]]*100:.2f}% — 보조손실이 무결손 성능을 해쳤는지")
    else:
        print(f"\n  Δ(jepa − recon) = {d:+.2f}pp   마진 ±{MARGIN_PP}pp   최대 조건내 시드폭 {spread:.2f}pp")
        if d >= 0:
            print("  판정: ★ 통과 — jepa 가 recon 이상 (특권 정보 없이 대등 이상)")
        elif abs(d) < MARGIN_PP:
            print("  판정: ★ 통과 — 마진 이내로 동등")
        else:
            print(f"  판정: ✗ 실패 — jepa 가 recon 에 {abs(d):.2f}pp 열세 (마진 초과)")
    return means


def gate23(data, degs, title, pred, label):
    print(f"\n{'='*78}\n{title} ({label})\n{'='*78}")
    print(f"  사전 예측: {pred}")
    for deg in degs:
        rows = defaultdict(dict)
        for (d, cond, seed), curve in data.items():
            if d == deg:
                rows[cond][seed] = curve
        if not rows:
            print(f"\n  [{deg}] 데이터 없음")
            continue
        print(f"\n  [{deg}]  {'조건':<9} {'n':>3}  {'AULC %':>8}  {'[min, max]':>16}  시드폭     L50")
        summ = {}
        for c in CONDS:
            if c not in rows:
                continue
            seeds = sorted(rows[c])
            a = [aulc(rows[c][s]) for s in seeds]
            ls = [l50(rows[c][s]) for s in seeds]
            lm = [x for x in ls if x is not None]
            l50s = f"{sum(lm)/len(lm):.2f}" if len(lm) == len(ls) else (
                   f">1.0 ({len(ls)-len(lm)}/{len(ls)}개 미붕괴)")
            summ[c] = (stat(a), a)
            print(f"        {c:<9} {len(seeds):>3}  {fmt_spread(a)}   {l50s}")

        # 판정: 조건간 평균 격차가 최대 조건내 시드폭을 넘어야만 신호로 인정한다.
        maxspread = max((s[0][2] - s[0][1]) for s in summ.values()) * 100
        order = sorted(summ, key=lambda c: -summ[c][0][0])
        gap = (summ[order[0]][0][0] - summ[order[-1]][0][0]) * 100
        print(f"        순위 {' > '.join(order)}   1위−3위 격차 {gap:.2f}pp   최대 시드폭 {maxspread:.2f}pp")
        if gap <= maxspread:
            print(f"        판정: 판정력 없음 — 격차가 시드 노이즈에 묻힘")
        else:
            j, r, s = (summ[c][0][0] * 100 if c in summ else float('nan') for c in CONDS)
            print(f"        판정: 신호 있음.  jepa−recon {j-r:+.2f}pp,  jepa−{CONDS[2]} {j-s:+.2f}pp")
            # 무겹침(가장 강한 증거): 두 조건의 시드 범위가 전혀 겹치지 않는가
            for a_, b_ in [("jepa", "recon"), ("jepa", CONDS[2])]:
                if a_ in summ and b_ in summ:
                    (_, alo, ahi), _ = summ[a_]
                    (_, blo, bhi), _ = summ[b_]
                    if alo > bhi:
                        print(f"              무겹침: {a_} 전 시드 > {b_} 전 시드 ★")
                    elif blo > ahi:
                        print(f"              무겹침: {b_} 전 시드 > {a_} 전 시드 ★")


def blind_report(data, label):
    """blind 대조군 — 시각 입력의 실제 기여분(무결손 → 완전 blind 낙폭)."""
    rows = {c: {s: cv for (d, cc, s), cv in data.items() if d == "blind" and cc == c}
            for c in CONDS}
    if not any(rows.values()):
        return
    print(f"\n{'='*78}\nblind 대조군 · 시각 기여분 ({label})\n{'='*78}")
    print("  좌표를 지형 무관 기준 점군으로 치환(valid=1 유지 → 분포 안, 지형 상호정보량 0)")
    print(f"\n  {'조건':<9} {'n':>3}  {'무결손 %':>9}  {'blind 1.0 %':>12}  {'낙폭 pp':>9}")
    drops = []
    for c in CONDS:
        if not rows[c]:
            continue
        seeds = sorted(rows[c])
        a = [rows[c][s][0][1] for s in seeds]
        b = [rows[c][s][-1][1] for s in seeds]
        d = [(x - y) * 100 for x, y in zip(a, b)]
        drops += d
        print(f"  {c:<9} {len(seeds):>3}  {sum(a)/len(a)*100:9.2f}  {sum(b)/len(b)*100:12.2f}"
              f"  {sum(d)/len(d):9.2f}")
    md = sum(drops) / len(drops)
    print(f"\n  평균 시각 기여분 = {md:.2f}pp")
    if md < 5:
        print("  판정: ★ 인코더가 과제에 거의 기여하지 않는다 — 인코더 초기화를 비교축으로 쓰는 한")
        print("        어떤 사전학습 목적함수도 신호를 낼 수 없다. 과제/지형을 바꿔야 한다.")
        print("        (Phase 3 무효 · 사전학습 우위 증발 · 게이트 3 무반응을 한 번에 설명)")
    elif md < 20:
        print("  판정: 시각 기여 약함 — 신호는 있으나 검정력이 빠듯하다. 지형 난이도 상향 필요.")
    else:
        print("  판정: ★ 시각은 과제에 본질적 — 게이트 3 무효는 시간 결손 설계 문제로 좁혀진다.")
        print("        (occlusion 이 부순 것은 정보 상실이 아니라 분포 밖 입력이라는 해석과 양립)")


def gate4(data, label):
    """게이트 4 — 정책이 ẑ 를 실제로 쓰는가(사전등록 v2 §4, 기전 검증).

    abl_curve_*.csv = 정책 입력의 ẑ 블록만 0 으로 치환하고 잰 level 0 성공률.
    기준선은 같은 런의 무결손 성능이며, 그 항등성은 게이트 1 이 이미 확인한다.
    """
    abl = {(c, s): cv[0][1] for (d, c, s), cv in data.items() if d == "abl"}
    if not abl:
        return
    base = {(c, s): cv[0][1] for (d, c, s), cv in data.items()
            if d != "abl" and cv[0][0] == 0.0}
    print(f"\n{'='*78}\n게이트 4 · 예측기가 실제로 쓰이는가 ({label})  — 기전 검증\n{'='*78}")
    print("  사전 예측: 낙폭(jepa) > 낙폭(none).  강인성은 간접 증거지만 이건 직접 측정이다.")
    print(f"\n  {'조건':<9} {'n':>3}  {'무결손 %':>9}  {'ẑ 절제 %':>10}  {'낙폭 pp':>9}  [min, max]")
    drops = {}
    for c in CONDS:
        seeds = sorted(s for (cc, s) in abl if cc == c and (cc, s) in base)
        if not seeds:
            continue
        b = [base[(c, s)] for s in seeds]
        a = [abl[(c, s)] for s in seeds]
        d = [(x - y) * 100 for x, y in zip(b, a)]
        drops[c] = d
        print(f"  {c:<9} {len(seeds):>3}  {sum(b)/len(b)*100:9.2f}  {sum(a)/len(a)*100:10.2f}"
              f"  {sum(d)/len(d):9.2f}  [{min(d):5.2f}, {max(d):5.2f}]")
    if "jepa" not in drops:
        return
    mj = sum(drops["jepa"]) / len(drops["jepa"])
    spread = max((max(v) - min(v)) for v in drops.values())
    print(f"\n  최대 조건내 시드폭 {spread:.2f}pp")
    for other in [c for c in CONDS if c != "jepa" and c in drops]:
        mo = sum(drops[other]) / len(drops[other])
        gap = mj - mo
        print(f"  낙폭(jepa) − 낙폭({other}) = {gap:+.2f}pp   → "
              f"{'★ 신호 있음' if gap > spread else '판정력 없음 — 시드 노이즈에 묻힘'}")
    if all(abs(sum(v) / len(v)) < 1.0 for v in drops.values()):
        print("\n  판정: 세 조건 모두 낙폭 ~0 — 예측기는 장식이고 v2 개입 자체가 실패다.")
        print("        사전등록 §4 에 따라 그대로 보고한다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--v2", action="store_true",
                    help="v2 비교축(jepa/recon/none · V2_* 런 · 시드폭 판정 · 게이트 4)")
    a = ap.parse_args()
    if a.v2:
        use_v2()
    for root in a.roots:
        label = "어려운 지형" if root.rstrip("/").endswith("_hard") else "기본 지형"
        data = load(root)
        if not data:
            print(f"\n{root}: CSV 없음 — 건너뜀")
            continue
        degs = sorted({d for d, _, _ in data})
        print(f"\n\n{'#'*78}\n# {root}  ({label})   결손 {degs}   커브 {len(data)}개\n{'#'*78}")
        gate1(data, label)
        gate23(data, [d for d in SPATIAL if d in degs],
               "게이트 2 · 공간 결손 강인성",
               "recon >= jepa" if V2 else "recon > scratch > jepa", label)
        gate23(data, [d for d in TEMPORAL if d in degs],
               "게이트 3 · 시간 결손 강인성  ★핵심",
               f"jepa > recon 및 jepa > {CONDS[2]}", label)
        blind_report(data, label)
        gate4(data, label)


if __name__ == "__main__":
    main()
