"""掃描專案文字檔,找出殘留的簡體字(以 OpenCC s2tw 偵測)。

用法:
    python src/check_traditional.py          # 只報告
    python src/check_traditional.py --fix     # 自動轉成繁體(僅就地修改 data/ 下的 .csv)

註:報告(.tex)與程式碼若出現 台/干/吃 等字,是「繁簡共用字」的誤報,本來就是正確
繁體,故列入白名單不報、也不轉(避免把「干擾」誤改成「幹擾」)。
"""
import sys
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parent.parent
cc = OpenCC("s2tw")  # 台灣標準:簡→繁

# 繁簡共用、在繁體中本就正確的字,OpenCC 仍會轉換,故排除以免誤報/誤改
WHITELIST = set("台干吃后里系制面松范谷板准周庄丑划")

TARGETS = [
    "report/報告.tex",
    "補充資料/課程資訊.md",
    "README.md",
    "data/測試資料集.csv",
    "data/測試題目.csv",
    "data/測試結果.csv",
    ".env.example",
    "requirements.txt",
]
TARGETS += [str(p.relative_to(ROOT)) for p in (ROOT / "src").glob("*.py")]


def simplified_chars(text: str) -> set:
    return {c for c in text if cc.convert(c) != c and c not in WHITELIST}


def main() -> None:
    fix = "--fix" in sys.argv
    issues = 0
    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        bad = simplified_chars(text)
        if not bad:
            continue
        issues += 1
        print(f"\n[簡體字] {rel}  → {len(bad)} 種: {''.join(sorted(bad))}")
        for i, line in enumerate(text.splitlines(), 1):
            if simplified_chars(line):
                print(f"   L{i}: {line.strip()[:80]}")
        # 只自動轉換 data/ 下的檔案(報告與程式碼維持手動把關)
        if fix and rel.startswith("data/"):
            path.write_text(cc.convert(text), encoding="utf-8")
            print("   ✓ 已用 s2tw 轉為繁體並覆寫")
        elif fix:
            print("   (非 data/ 檔,未自動轉換;請人工確認)")

    if issues == 0:
        print("✓ 所有檔案皆為繁體中文,未發現簡體字。")


if __name__ == "__main__":
    main()
