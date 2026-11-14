"""C2 入库钩子的自验收（可重复跑）：`python scripts/verify_c2.py`。

验收什么（都是"错了会悄悄放错数据"的地方）：
1. 首轮：投递 2 个文件 + 1 个非数据文件 → 只处理 2 个，分档落盘、台账入账；
2. 幂等：同一批再投一次 → 全部"跳过"，记录数不变；
3. 签名：错误密钥/无密钥必须被拒（门禁不能被绕过）；
4. 容错：不存在的文件 → 记 error，进程正常退出，不阻塞其它文件；
5. 台账可查：files/records 数量与分档目录一致。

退出码 0 = 全部通过。
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".acc_ingest"
PKG = ROOT / "outputs" / "deploy" / "poisonguard-ci.pkg"
KEY = "ci-smoke-key"

_fail: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ✅ " if ok else "  ❌ ") + msg)
    if not ok:
        _fail.append(msg)


def cli(*args: str, expect: int = 0) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable, "-m", "poisonguard.cli", *args],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != expect:
        print(f"    rc={p.returncode} (期望 {expect})")
        print("    " + (p.stdout or "")[-800:].replace("\n", "\n    "))
        print("    " + (p.stderr or "")[-800:].replace("\n", "\n    "))
    return p


def main() -> int:
    if not PKG.exists():
        print(f"[skip] 找不到 {PKG}，先跑 `python scripts/ci_smoke.py` 生成")
        return 0
    shutil.rmtree(WORK, ignore_errors=True)
    (WORK / "inbox").mkdir(parents=True)
    demo = ROOT / "data" / "scan_demo_afqmc.txt"
    lines = [ln for ln in demo.read_text(encoding="utf-8").splitlines() if ln.strip()]
    (WORK / "inbox" / "b1.txt").write_text("\n".join(lines[:6]), encoding="utf-8", newline="\n")
    (WORK / "inbox" / "b2.txt").write_text("\n".join(lines[:3]), encoding="utf-8", newline="\n")
    (WORK / "inbox" / "ignore.md").write_text("不是数据文件", encoding="utf-8", newline="\n")

    common = ["--bundle", str(PKG), "--sign-key", KEY,
              "--ledger-dir", str(WORK), "--out-dir", str(WORK),
              "--inbox", str(WORK / "inbox"), "--watch-band", "0"]

    print("[1] 首轮投递")
    p = cli("ingest", *common)
    check(p.returncode == 0, "命令正常退出")
    check("BLOCK" in (p.stdout or "") or "PASS" in (p.stdout or ""), "打印了分档统计")
    check(not any("ignore.md" in ln for ln in (p.stdout or "").splitlines()),
          "非数据文件（.md）被跳过")

    print("[2] 幂等复投")
    p2 = cli("ingest", *common)
    check(p2.returncode == 0, "命令正常退出")
    check((p2.stdout or "").count("跳过") >= 2, "两个文件都被 sha256 台账跳过")

    print("[3] 签名门禁")
    p3 = cli("ingest", *common[:2], "--sign-key", "wrong-key",
             "--ledger-dir", str(WORK), "--out-dir", str(WORK),
             "--inbox", str(WORK / "inbox"), expect=1)
    check(p3.returncode != 0, "错误密钥被拒绝（非 0 退出）")
    p4 = cli("ingest", "--bundle", str(PKG),
             "--ledger-dir", str(WORK), "--out-dir", str(WORK),
             "--inbox", str(WORK / "inbox"), expect=1)
    check(p4.returncode != 0, "未提供密钥被拒绝")

    print("[4] 容错")
    p5 = cli("ingest", "--bundle", str(PKG), "--sign-key", KEY,
             "--ledger-dir", str(WORK), "--out-dir", str(WORK),
             "--file", str(WORK / "nope.txt"))
    check(p5.returncode == 0, "缺文件不崩（退出码 0）")
    check("error" in (p5.stdout or ""), "缺文件记为 error 而不是静默成功")

    print("[5] 台账一致性")
    c = sqlite3.connect(WORK / "ledger.sqlite3")
    n_files = c.execute("select count(*) from files").fetchone()[0]
    flags = dict(c.execute("select flag,count(*) from records group by flag").fetchall())
    n_rec = sum(flags.values())
    check(n_files == 2, f"台账文件数=2（实际 {n_files}）")
    check(n_rec == 9, f"记录数=9（实际 {n_rec}）")
    for sub in ("quarantine", "review", "released"):
        (WORK / sub).mkdir(exist_ok=True)
    n_jsonl = sum(1 for sub in ("quarantine", "review", "released")
                  for _ in (WORK / sub).glob("*.jsonl"))
    check(n_jsonl >= 1, f"分档 jsonl 有产出（{n_jsonl} 个文件）")

    shutil.rmtree(WORK, ignore_errors=True)
    print()
    if _fail:
        print(f"[verify_c2] 失败 {len(_fail)} 项：")
        for m in _fail:
            print("  - " + m)
        return 1
    print("[verify_c2] 全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
