#!/usr/bin/env python3
"""R0-3 独立复算：DeepMind OP3 双足足球论文 Table 1 与 Fig 7 聚合值。

数据源：Zenodo record 10793725 (op3_soccer.zip)，只读解包到工作目录。
本脚本为独立实现（不 import / 不执行作者 notebook 代码），仅共享
(1) 发布的原始逐 timestep 数据，
(2) 论文正文声明的实验口径（arXiv 2304.13653v2）。
常数出处见各处注释，全部给出论文行号/页码依据。

用法：python3 r03_recompute_soccer.py
退出码：0 = 全部计算完成（与论文值是否一致不影响退出码）。
"""
import base64
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = "/Users/guohongbin/1-Projects/duckbench-rir"
ZIP_PATH = f"{REPO}/sources/soccer-deepmind/op3_soccer.zip"
NB_FIG7 = "Fig7-S3-S4.ipynb"  # zip 内成员名
WORK = "/tmp/r03-work/extracted"  # 只读解包目标
DATA = f"{WORK}/op3_soccer/data_strings"

# ---------------- 论文声明的口径常数（arXiv 2304.13653v2） ----------------
# 球门开口宽 0.8 m（paper.txt L568: "two goals that each have an opening width of 0.8 m"）
GOAL_WIDTH_PAPER = 0.80
# 作者 notebook 用 goal_width = 0.85（nb cell "Environment and analysis settings"）——做敏感性对照
GOAL_WIDTH_NOTEBOOK = 0.85
GOAL_HEIGHT = 0.5          # notebook 常数；论文未给出球门高度数值
UPRIGHT_Z = 0.36           # 36cm 肩部 mocap marker（paper L1849-1850，完全直立 41cm）
STEADY_SECONDS = 1.0       # 须保持直立时长（nb 常数；论文 L1849 "as soon as it reached 36cm"，未声明稳定窗）
CONTACT_THRESH = 0.01      # 球移动 >1cm 视为触球（nb 常数；论文未给数值）
KICK_WINDOW = 0.2          # 触球后 0.2s 位移测踢速（paper L1866-1867）
TURN_LO, TURN_HI = np.pi / 4, 3 * np.pi / 4  # ±45°→±135°（paper L1877-1878）
WALK_T0, WALK_T1, SUBPATH = 2.0, 7.0, 1.0    # 第 2s→7s，按 1s 子窗求和（paper L1835-1839）
NUM_WALK_SECONDS = 5.0
PITCH_X_LIMIT = 2.5        # 球门线 x=±2.5（场地 5m 长，paper L77/L568）

SUBSTEP = {  # 各实验 timestep 时长（nb 常数 cell；与数据 Time 列实际步长交叉验证）
    "get_up_and_shoot": 0.1,
    "get_up": 0.025,
    "walking_speed": 0.025,
    "kicking_power": 0.025,
    "turning_speed": 0.025,
}

# 论文 Table 1 报告值（pdftotext paper.txt L162-237 手工转录；括号内为 std err）
PAPER_TABLE1A = {  # Get-up-and-shoot set piece
    ("Scoring Success Rate", "sim"): (0.70, 0.07),
    ("Scoring Success Rate", "real"): (0.58, 0.07),
    ("Time to Ball Contact", "sim"): (4.6, 0.16),
    ("Time to Ball Contact", "real"): (4.7, 0.14),
}
PAPER_TABLE1B = {  # (controller, stat)
    ("real_baseline", "Walking Speed"): (0.20, 0.005),
    ("real_learned", "Walking Speed"): (0.57, 0.003),
    ("sim_learned", "Walking Speed"): (0.51, 0.01),
    ("real_baseline", "Turning Speed"): (0.71, 0.04),
    ("real_learned", "Turning Speed"): (2.85, 0.19),
    ("sim_learned", "Turning Speed"): (3.19, 0.12),
    ("real_baseline", "Get-up Time"): (2.52, 0.006),
    ("real_learned", "Get-up Time"): (0.93, 0.12),
    ("sim_learned", "Get-up Time"): (0.73, 0.01),
    ("real_baseline", "Kicking Speed"): (2.07, 0.05),
    ("real_learned", "Kicking Speed"): (2.02, 0.26),
    ("real_with_run_up", "Kicking Speed"): (2.77, 0.11),
    ("sim_learned", "Kicking Speed"): (2.12, 0.07),
}
# 论文正文（摘要 + L253-256）：181% faster / 302% faster / 63% less / 34% faster
PAPER_IMPROVEMENT = {"Walking Speed": 181, "Turning Speed": 302,
                     "Get-up Time": 63, "Kicking Speed": 34}

RESULTS = {"table1a": {}, "table1b": {}, "improvements": {},
           "sensitivity": {}, "fig7": {}, "checks": {}}


def ensure_extracted():
    if os.path.isdir(DATA):
        return
    os.makedirs(WORK, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(WORK)


def load_exp(exp):
    """读一个实验目录全部 base64-parquet txt，返回 {name: DataFrame}。"""
    d = os.path.join(DATA, exp)
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".txt"):
            raw = open(os.path.join(d, fn)).read()
            out[fn[: -len(".txt")]] = pd.read_parquet(io.BytesIO(base64.b64decode(raw)))
    return out


def episodes(df):
    """DataFrame -> list of np.ndarray [T,4]（Time, c3, c4, c5），按 Trial 切分。"""
    arr = df.to_numpy()
    n = int(arr[:, 0].max()) + 1
    return [arr[arr[:, 0] == k][:, 2:] for k in range(n)]


def stats(vals):
    """mean / 样本标准差(ddof=1) / SEM（作者口径：SEM = std/sqrt(n)）。"""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "std": std, "sem": sem}


# ---------------- 我的独立实现（与 notebook 代码无共享） ----------------
def goal_scored(ball, goal_width=GOAL_WIDTH_NOTEBOOK, goal_height=GOAL_HEIGHT):
    """任意时刻球位于 x<=-2.5（左）或 x>=+2.5（右）且 |y|<w/2 且 z<h → 进球。
    我的实现：两端都检查并报告进的是哪侧（作者默认只查 'left'）。"""
    x, y, z = ball[:, 1], ball[:, 2], ball[:, 3]
    left = (x <= -PITCH_X_LIMIT) & (np.abs(y) < goal_width / 2) & (z < goal_height)
    right = (x >= PITCH_X_LIMIT) & (np.abs(y) < goal_width / 2) & (z < goal_height)
    return 1 if (left.any() or right.any()) else 0, ("left" if left.any() else
                                                     "right" if right.any() else "none")


def time_first_touch(ball):
    """球首次离开初始位置 >1cm 的前一时刻；球不动返回 inf（同作者口径，含警告计数）。"""
    init = ball[0, 1:]
    d = np.linalg.norm(ball[:, 1:] - init, axis=1)
    hit = np.where(d > CONTACT_THRESH)[0]
    if len(hit) == 0:
        return float("inf"), -1
    idx = max(int(hit[0]) - 1, 0)  # 作者实现此处可为 -1；我钳到 0，差异见 checks
    return float(ball[idx, 0]), idx


def get_up_time(player, substep):
    """首次满足：此后 ceil(1.0/substep) 步连续 z>0.36cm 的时刻。"""
    k = int(np.ceil(STEADY_SECONDS / substep))
    up = player[:, 3] > UPRIGHT_Z
    T = len(up)
    for i in range(T - k + 1):
        if up[i:i + k].all():
            return float(player[i, 0])
    return None  # 作者返回 None -> np.mean 崩溃；此处保留 None 并计数


def walk_speed(player):
    """t∈[2,7]s，5 个 1s 子窗位移求和 / 5s（paper L1835-1839）。"""
    t = player[:, 0]
    total_d, total_t = 0.0, 0.0
    for i in range(int(NUM_WALK_SECONDS / SUBPATH)):
        i0 = int(np.argmin(np.abs(t - (WALK_T0 + i * SUBPATH))))
        i1 = int(np.argmin(np.abs(t - (WALK_T0 + (i + 1) * SUBPATH))))
        total_d += float(np.linalg.norm(player[i1, 1:3] - player[i0, 1:3]))
        total_t += float(t[i1] - t[i0])
    return total_d / total_t


def kick_speed(ball, substep):
    """触球后 0.2s 窗口内球位移 / 0.2s（paper L1866-1867）。"""
    _t, c = time_first_touch(ball)
    if c < 0:
        return float("inf")
    j = c + int(np.ceil(KICK_WINDOW / substep))
    if j >= len(ball):
        j = len(ball) - 1
    dt = float(ball[j, 0] - ball[c, 0])
    return float(np.linalg.norm(ball[j, 1:] - ball[c, 1:]) / dt)


def turn_speed(euler):
    """|yaw| 从 π/4 转到 3π/4 的用时求角速度（paper L1877-1878）。
    euler 列: [Time, yaw, roll, pitch]（发布文件列名 'X pos'... 为误标，见 checks）。"""
    yaw = euler[:, 1]
    hi_cw = np.where(yaw > TURN_HI)[0]
    hi_ccw = np.where(yaw < -TURN_HI)[0]
    if len(hi_cw) == 0 and len(hi_ccw) == 0:
        return None
    cw = len(hi_ccw) == 0 or (len(hi_cw) > 0 and hi_cw[0] < hi_ccw[0])
    if cw:
        lo = np.where(yaw > TURN_LO)[0]
        finish_t = float(euler[hi_cw[0], 0])
    else:
        lo = np.where(yaw < -TURN_LO)[0]
        finish_t = float(euler[hi_ccw[0], 0])
    if len(lo) == 0:
        return None
    start_t = float(euler[max(lo[0] - 1, 0), 0])
    return abs((TURN_HI - TURN_LO) / (finish_t - start_t))


# ---------------- Table 1 复算 ----------------
def recompute_table1():
    # --- 1a: get-up-and-shoot ---
    gu = load_exp("get_up_and_shoot")
    for env in ("sim", "real"):
        balls = episodes(gu[f"{env}_get_up_and_shoot_ball_tstxyz_df"])
        players = episodes(gu[f"{env}_get_up_and_shoot_player_tstxyz_df"])
        assert len(balls) == len(players)
        goals, sides, contact, no_goal_touch = [], [], [], 0
        for b, p in zip(balls, players):
            g, side = goal_scored(b)
            sides.append(side)
            goals.append(g)
            t, _ = time_first_touch(b)
            if np.isinf(t):
                no_goal_touch += 1
            contact.append(t)
        st_rate = stats(goals)
        st_touch_all = stats(contact)
        finite = [t for t in contact if np.isfinite(t)]
        st_touch_fin = stats(finite) if len(finite) else {"n": 0, "mean": None,
                                                          "std": None, "sem": None}
        key = {"sim": "sim", "real": "real"}[env]
        RESULTS["table1a"][key] = {
            "scoring_success_rate": st_rate,
            "time_to_ball_contact_incl_inf": st_touch_all,
            "time_to_ball_contact_excl_inf": st_touch_fin,
            "goal_side_counts": {s: sides.count(s) for s in set(sides)},
            "substep_s": SUBSTEP["get_up_and_shoot"],
        }
        RESULTS["checks"][f"guas_{env}_no_touch_episodes"] = no_goal_touch

    # --- 1b ---
    # Walking Speed
    ws = load_exp("walking_speed")
    for ctrl in ("real_baseline", "real_learned", "sim_learned"):
        speeds = [walk_speed(e) for e in episodes(ws[f"{ctrl}_walking_speed_player_tstxyz_df"])]
        RESULTS["table1b"][(ctrl, "Walking Speed")] = stats(speeds)

    # Turning Speed（euler 文件：列 Time,yaw,roll,pitch）
    ts = load_exp("turning_speed")
    for ctrl in ("real_baseline", "real_learned", "sim_learned"):
        vels = []
        for e in episodes(ts[f"{ctrl}_turning_speed_euler_tstyrp_df"]):
            v = turn_speed(e)
            vels.append(v if v is not None else float("nan"))
        s = stats(vels)
        s["failed_turns"] = int(np.sum(np.isnan(np.asarray(vels, float))))
        RESULTS["table1b"][(ctrl, "Turning Speed")] = s

    # Get-up Time
    gt = load_exp("get_up")
    for ctrl in ("real_baseline", "real_learned", "sim_learned"):
        times, fails = [], 0
        for e in episodes(gt[f"{ctrl}_get_up_player_tstxyz_df"]):
            t = get_up_time(e, SUBSTEP["get_up"])
            if t is None:
                fails += 1
            else:
                times.append(t)
        s = stats(times)
        s["never_upright_episodes"] = fails
        RESULTS["table1b"][(ctrl, "Get-up Time")] = s

    # Kicking Speed
    kp = load_exp("kicking_power")
    for ctrl in ("real_baseline", "real_learned", "real_with_run_up", "sim_learned"):
        vels = [kick_speed(e, SUBSTEP["kicking_power"])
                for e in episodes(kp[f"{ctrl}_kicking_power_ball_tstxyz_df"])]
        RESULTS["table1b"][(ctrl, "Kicking Speed")] = stats(vels)

    # --- 改善百分比（同作者口径：Kicking 用 with_run_up，其余用 real_learned；Get-up 用 less=100*(1-ratio)） ---
    for stat, (num_ctrl) in (("Walking Speed", "real_learned"),
                             ("Turning Speed", "real_learned"),
                             ("Get-up Time", "real_learned"),
                             ("Kicking Speed", "real_with_run_up")):
        num = RESULTS["table1b"][(num_ctrl, stat)]["mean"]
        den = RESULTS["table1b"][("real_baseline", stat)]["mean"]
        RESULTS["improvements"][stat] = (100 * (num / den - 1) if stat != "Get-up Time"
                                         else 100 * (1 - num / den))


# ---------------- 敏感性 / 反证检查 ----------------
def sensitivity():
    gu = load_exp("get_up_and_shoot")
    for env in ("sim", "real"):
        balls = episodes(gu[f"{env}_get_up_and_shoot_ball_tstxyz_df"])
        for gw in (GOAL_WIDTH_PAPER, GOAL_WIDTH_NOTEBOOK):
            flags = [goal_scored(b, goal_width=gw)[0] for b in balls]
            RESULTS["sensitivity"][f"success_rate_{env}_goal_width_{gw}"] = stats(flags)
    # timestep 步长与声明常数交叉验证
    for exp, key in (("get_up_and_shoot", "get_up_and_shoot"),
                     ("walking_speed", "walking_speed")):
        d = load_exp(exp)
        name = sorted(d)[0]
        arr = d[name].to_numpy()
        RESULTS["checks"][f"dt_{exp}"] = float(arr[1, 2] - arr[0, 2])
    # turn 文件列名与语义
    tsd = load_exp("turning_speed")
    cols = list(tsd.values())[0].columns
    RESULTS["checks"]["turning_cols"] = [str(c) for c in cols]
    # kicking: 首帧即触球的边界情况（作者 first_contact_ind=-1 缺陷是否触发）
    kp = load_exp("kicking_power")
    n_edge = 0
    for name, df in kp.items():
        for e in episodes(df):
            if np.linalg.norm(e[0, 1:] - e[1, 1:]) > CONTACT_THRESH or \
               float(np.linalg.norm(e[0, 1:4] - e[0, 1:4])) >= 0:  # 占位
                pass
            d0 = np.linalg.norm(e[0, 1:] - e[0, 1:])
            # 真正检查：第0步到第1步球位移 >1cm 说明首帧即动
            if np.linalg.norm(e[1, 1:] - e[0, 1:]) > CONTACT_THRESH and \
               np.linalg.norm(e[2, 1:] - e[1, 1:]) > CONTACT_THRESH:
                n_edge += 1
    RESULTS["checks"]["kick_first_frame_moving_suspect"] = n_edge


# ---------------- Fig 7（作者发布的预聚合数据） ----------------
def recompute_fig7():
    zpath = ZIP_PATH
    with zipfile.ZipFile(zpath) as z:
        nb = z.read(f"op3_soccer/{NB_FIG7}").decode("utf-8")
    cells = json.loads(nb)["cells"]
    strs = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for tag in ("df_fig7_bytestr", "df_fig10_bytestr", "df_fig11_bytestr"):
            if f"{tag} = '" in src:
                s = src.split(f"{tag} = '")[1].split("'")[0]
                strs.append((tag, s))
    for tag, s in strs:
        df = pd.read_parquet(io.BytesIO(base64.b64decode(s)))
        if tag == "df_fig7_bytestr":
            out = {}
            for abl, g in df.groupby("ablation_name"):
                g = g.sort_values("ten_million_actor_steps")
                last = g.iloc[-1]
                out[abl] = {
                    "n_points": int(len(g)),
                    "x_last_1e7": float(last["ten_million_actor_steps"]),
                    "y_mean_last": float(last["y_mean"]),
                    "y_stderr_last": float(last["y_stderr"]),
                    "ci2_last": float(last["y_stderr"]) * 2,
                    "ci_t_last": float(last["y_stderr"]) * 2.776,  # t(4,.975)，若 5 seeds
                    "y_mean_max": float(g["y_mean"].max()),
                }
            RESULTS["fig7"]["curves_final"] = out
            RESULTS["fig7"]["columns"] = list(df.columns)
            # 论文定性论断核对：full self-play (A) > fixed opponents (B)；oppfrac10 (C) 更差
            a = out.get("self-play", {}).get("y_mean_last")
            b = out.get("opponents6", {}).get("y_mean_last")
            c3 = out.get("self-play_oppfrac10", {}).get("y_mean_last")
            RESULTS["fig7"]["paper_claims_check"] = {
                "A_full_ge_B_fixed": bool(a >= b) if a and b else None,
                "A_vs_B_diff": float(a - b) if a and b else None,
                "A_ge_C_oppfrac10": bool(a >= c3) if a and c3 else None,
            }


def print_report():
    def f(x, d=4):
        return ("%.4f" % x) if isinstance(x, (int, float)) and x is not None else str(x)

    print("=" * 88)
    print("TABLE 1a  Get-up-and-Shoot（分母 = 全部 episode，无排除）")
    print("=" * 88)
    for env in ("sim", "real"):
        for stat, pk in (("scoring_success_rate", ("Scoring Success Rate", env)),
                         ("time_to_ball_contact_excl_inf", ("Time to Ball Contact", env))):
            r = RESULTS["table1a"][env][stat]
            pv, pe = PAPER_TABLE1A[pk]
            use = r["mean"] if stat == "scoring_success_rate" else r["mean"]
            print(f"[{env}] {stat:32s} paper={pv}({pe})  mine={f(use)}"
                  f" ±{f(r['sem'])}  n={r['n']}  diff={f(use - pv)}")
    for env in ("sim", "real"):
        print(f"  goal sides {env}: {RESULTS['table1a'][env]['goal_side_counts']}  "
              f"no-touch episodes: {RESULTS['checks'][f'guas_{env}_no_touch_episodes']}")

    print("=" * 88)
    print("TABLE 1b  基线对比（分母：各实验 10 episodes；turning 论文声明 13 取 10）")
    print("=" * 88)
    rows = [("real_baseline", "Walking Speed"), ("real_learned", "Walking Speed"),
            ("sim_learned", "Walking Speed"), ("real_baseline", "Turning Speed"),
            ("real_learned", "Turning Speed"), ("sim_learned", "Turning Speed"),
            ("real_baseline", "Get-up Time"), ("real_learned", "Get-up Time"),
            ("sim_learned", "Get-up Time"), ("real_baseline", "Kicking Speed"),
            ("real_learned", "Kicking Speed"), ("real_with_run_up", "Kicking Speed"),
            ("sim_learned", "Kicking Speed")]
    for key in rows:
        pv, pe = PAPER_TABLE1B[key]
        r = RESULTS["table1b"][key]
        print(f"{key[0]:16s} {key[1]:14s} paper={pv}({pe})  mine={f(r['mean'])}"
              f" ±{f(r['sem'])}  n={r['n']}  diff={f(r['mean'] - pv)}"
              + (f"  failed_turns={r['failed_turns']}" if "failed_turns" in r and r["failed_turns"] else "")
              + (f"  never_up={r['never_upright_episodes']}" if r.get("never_upright_episodes") else ""))
    print("-" * 88)
    print("Improvement vs paper text:")
    for stat, pv in PAPER_IMPROVEMENT.items():
        print(f"  {stat:14s} paper={pv}%  mine={f(RESULTS['improvements'][stat], 2)}%")
    print("=" * 88)
    print("SENSITIVITY: goal width 0.85 (notebook) vs 0.80 (paper text)")
    for k, v in RESULTS["sensitivity"].items():
        print(f"  {k:52s} mean={f(v['mean'])} n={v['n']}")
    print("=" * 88)
    print("FIG 7 (作者预聚合 parquet；无 per-seed/per-match 原始数据，复算限于聚合核对)")
    for abl, v in RESULTS["fig7"].get("curves_final", {}).items():
        print(f"  {abl:34s} x_last={v['x_last_1e7']}e7  y_mean={f(v['y_mean_last'])}"
              f" ±2SEM={f(v['ci2_last'])} (tCI={f(v['ci_t_last'])}) pts={v['n_points']}")
    print("  claims:", json.dumps(RESULTS["fig7"].get("paper_claims_check", {})))
    print("=" * 88)
    print("CHECKS:", json.dumps(RESULTS["checks"], default=str, indent=2))


def main():
    ensure_extracted()
    recompute_table1()
    sensitivity()
    recompute_fig7()
    print_report()
    with open(f"{REPO}/r0/r03-scripts/r03_results.json", "w") as f:
        RESULTS_S = {k: ({f"{a}|{b}": v for (a, b), v in val.items()}
                         if k == "table1b" else val)
                     for k, val in RESULTS.items()}
        json.dump(RESULTS_S, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
