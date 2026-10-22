#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小红书搜索语料机械分析（纯标准库，无需安装依赖）

输入：一行一条笔记标题，可选在标题后用 TAB 或 2 个以上空格接点赞数。
      例：五年级期中家长会PPT｜班主任直接复制   1.2万
输出：标题结构统计 + 高频 n-gram（候选长尾词）+ 高频开头/结尾 + 句式分类。

用法：
  python analyze_corpus.py 语料.txt
  python analyze_corpus.py 语料.txt --out 报告.md --top 60 --minfreq 2
  python analyze_corpus.py 语料.txt --json

注意：脚本只做统计，不做语义判断。六维意图拆解、选题排序由 Skill 主流程完成。
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter

CJK = re.compile(r"[\u4e00-\u9fff]")
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\u2190-\u21FF\u2B00-\u2BFF]"
)
# 出现在词首或词尾时基本可以判定为切词噪声的字
EDGE_NOISE = set(
    "的了呢吗吧啊么呀嘛和与及或是在有我你他她它这那哪什怎就都也很还个套份次种些把被给让对从向于为以而但并且如果"
    "级届才最直能完分不就再更太挺得地着过来去说做"
)

SENTENCE_PATTERNS = [
    ("疑问式", re.compile(r"[?？]|吗$|呢$|怎么|如何|怎样|为什么|到底|哪个|哪些")),
    ("数字清单式", re.compile(r"^\d|\d+\s*[个套份种步招条篇]|第[一二三四五六七八九十]|TOP|Top|top\d")),
    ("时间节点式", re.compile(r"20\d\d|\d+月|期中|期末|开学|寒假|暑假|月考|季|最新|新版|版$")),
    ("人群指向式", re.compile(r"一年级|二年级|三年级|四年级|五年级|六年级|初一|初二|初三|高一|高二|高三|年级|班主任|老师|家长|宝妈|大学生|新手|小白|零基础")),
    ("对比取舍式", re.compile(r"vs|VS|对比|区别|还是|不如|别再|劝你|不要|避雷|踩坑")),
    ("结果承诺式", re.compile(r"一次|直接|搞定|省时|省力|保姆级|无脑|照抄|复制|模板|全套|免费|不用|秒|稳")),
    ("情绪钩子式", re.compile(r"绝了|真的|终于|居然|后悔|救命|谁懂|必看|收藏|🔥|爆|哭")),
]


def parse_line(raw):
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    line = re.sub(r"^[-*•·]\s*", "", line)  # markdown 列表
    if line.startswith("|"):
        return None
    likes = None
    m = re.search(r"(?:\t| {2,})[^\d]{0,2}(\d+(?:\.\d+)?)\s*(万|w|W)?\s*$", line)
    if m:
        val = float(m.group(1))
        if m.group(2):
            val *= 10000
        likes = int(val)
        line = line[: m.start()].strip()
    line = line.strip(" \t　")
    if not CJK.search(line):
        return None
    return {"title": line, "likes": likes}


def cjk_only(s):
    return "".join(CJK.findall(s))


def ngrams(texts, min_n=2, max_n=6):
    counter = Counter()
    for t in texts:
        c = cjk_only(t)
        L = len(c)
        for n in range(min_n, max_n + 1):
            for i in range(L - n + 1):
                g = c[i : i + n]
                if g[0] in EDGE_NOISE or g[-1] in EDGE_NOISE:
                    continue
                if len(set(g)) == 1:  # 叠字噪声
                    continue
                counter[g] += 1
    return counter


def prune(counter, minfreq):
    """去掉被更长且频次不低于自己的片段完全覆盖的短片段。"""
    items = {g: f for g, f in counter.items() if f >= minfreq}
    result = {}
    for g, f in items.items():
        covered = False
        for h, fh in items.items():
            if len(h) > len(g) and g in h and fh >= f:
                covered = True
                break
        if not covered:
            result[g] = f
    return Counter(result)


def edge_fragments(titles, k=4, minfreq=2):
    head, tail = Counter(), Counter()
    for t in titles:
        c = cjk_only(t)
        if len(c) >= k:
            head[c[:k]] += 1
            tail[c[-k:]] += 1
    head = Counter({g: f for g, f in head.items() if f >= minfreq})
    tail = Counter({g: f for g, f in tail.items() if f >= minfreq})
    return head, tail


def classify(titles):
    rows = []
    for name, pat in SENTENCE_PATTERNS:
        hits = [t for t in titles if pat.search(t)]
        rows.append((name, len(hits), round(len(hits) / max(len(titles), 1) * 100, 1)))
    rows.sort(key=lambda r: -r[1])
    return rows


def length_stats(titles):
    # 按整条标题字符数计（含字母数字），对应小红书 20 字上限的口径
    lens = [len(t.replace(" ", "").replace("　", "")) for t in titles if t.strip()]
    if not lens:
        return {}
    return {
        "中位字数": statistics.median(lens),
        "均值": round(statistics.mean(lens), 1),
        "最短": min(lens),
        "最长": max(lens),
        "20字以内占比": round(sum(1 for x in lens if x <= 20) / len(lens) * 100, 1),
    }


def shape_features(titles):
    n = max(len(titles), 1)
    feats = {
        "含数字": sum(1 for t in titles if re.search(r"\d", t)),
        "含竖线或书名号": sum(1 for t in titles if re.search(r"[|｜【】\[\]]", t)),
        "含emoji": sum(1 for t in titles if EMOJI.search(t)),
        "含标点断句": sum(1 for t in titles if re.search(r"[，,。！!、]", t)),
        "含年份": sum(1 for t in titles if re.search(r"20\d\d", t)),
        "含可编辑/可下载承诺": sum(1 for t in titles if re.search(r"可编辑|可改|可下载|可打印|源文件|无水印", t)),
        "含英文缩写": sum(1 for t in titles if re.search(r"[A-Za-z]{2,}", t)),
    }
    return {k: f"{v} ({round(v / n * 100, 1)}%)" for k, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", default=None, help="输出 Markdown 报告路径")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--minfreq", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    raw = open(args.corpus, encoding="utf-8", errors="ignore").read().splitlines()
    rows = [r for r in (parse_line(x) for x in raw) if r]
    if not rows:
        print("语料为空，检查文件编码或格式", file=sys.stderr)
        sys.exit(1)

    titles = [r["title"] for r in rows]
    likes = [r["likes"] for r in rows if r["likes"] is not None]

    gram = prune(ngrams(titles), args.minfreq)
    head, tail = edge_fragments(titles)
    cls = classify(titles)

    top_titles = sorted(
        [r for r in rows if r["likes"] is not None], key=lambda r: -r["likes"]
    )[:10]

    # 点赞前 25% 的标题长度区间，作为该词下的字数参考
    head_len = None
    if len(likes) >= 8:
        ranked = sorted([r for r in rows if r["likes"] is not None], key=lambda r: -r["likes"])
        q = ranked[: max(3, len(ranked) // 4)]
        ql = sorted(len(r["title"].replace(" ", "")) for r in q)
        head_len = {"样本数": len(ql), "字数区间": f"{ql[0]}-{ql[-1]}", "中位": statistics.median(ql)}

    payload = {
        "样本数": len(titles),
        "带点赞数据": len(likes),
        "点赞中位数": statistics.median(likes) if likes else None,
        "标题字数": length_stats(titles),
        "高互动标题字数": head_len,
        "结构特征": shape_features(titles),
        "句式分布": [{"类型": a, "条数": b, "占比%": c} for a, b, c in cls],
        "高频片段": [
            {"片段": g, "频次": f}
            for g, f in sorted(gram.items(), key=lambda kv: -(kv[1] * len(kv[0]) ** 0.8))[: args.top]
        ],
        "高频开头4字": [{"片段": g, "频次": f} for g, f in head.most_common(15)],
        "高频结尾4字": [{"片段": g, "频次": f} for g, f in tail.most_common(15)],
        "高互动样本": [
            {"标题": r["title"], "点赞": r["likes"]} for r in top_titles
        ],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    lines = []
    lines.append(f"# 语料分析：{args.corpus}\n")
    lines.append(f"样本 {len(titles)} 条，其中 {len(likes)} 条带点赞数据")
    ls = payload["标题字数"]
    if ls:
        lines.append(
            f"标题字数中位数 {ls['中位字数']}，均值 {ls['均值']}，"
            f"范围 {ls['最短']}-{ls['最长']}，20 字以内占 {ls['20字以内占比']}%"
        )
    hl = payload["高互动标题字数"]
    if hl:
        lines.append(f"点赞前 25% 的 {hl['样本数']} 条标题：字数 {hl['字数区间']}，中位 {hl['中位']}")
    lines.append("")
    lines.append("## 结构特征")
    for k, v in payload["结构特征"].items():
        lines.append(f"- {k}：{v}")
    lines.append("")
    lines.append("## 句式分布")
    for r in payload["句式分布"]:
        lines.append(f"- {r['类型']}：{r['条数']} 条（{r['占比%']}%）")
    lines.append("")
    lines.append("## 高频片段（候选长尾词，需人工判义）")
    for r in payload["高频片段"]:
        lines.append(f"- {r['片段']} ×{r['频次']}")
    lines.append("")
    lines.append("## 高频开头 / 结尾（频次≥2）")
    for label, key in (("开头", "高频开头4字"), ("结尾", "高频结尾4字")):
        if payload[key]:
            lines.append(label + "：" + "、".join(f"{r['片段']}({r['频次']})" for r in payload[key]))
        else:
            lines.append(f"{label}：样本不足，无重复片段")
    if top_titles:
        lines.append("")
        lines.append("## 高互动样本")
        for r in top_titles:
            lines.append(f"- {r['likes']:>7} | {r['title']}")
    text = "\n".join(lines)

    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")
        print(f"已写入 {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
