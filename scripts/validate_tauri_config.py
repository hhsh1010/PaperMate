"""离线校验 tauri.conf.json 是否符合 Tauri 2 官方 schema。

为什么需要
----------
`tauri-build` 的 serde 报错**只报第一个**（`missing field 'type'`），
修完一个再跑才知道下一个 —— 一轮一个错，很折腾。
而且 serde 的报错文案对 tagged enum 很不友好
（把「键名写错」报成「缺 type 字段」，让人以为是结构问题）。

这里用官方 schema（https://schema.tauri.app/config/2）做**离线全量校验**，
一次列出全部不符合项。只实现本项目用到的关键字子集，够用即可。

用法：
    .venv/Scripts/python.exe scripts/validate_tauri_config.py
    .venv/Scripts/python.exe scripts/validate_tauri_config.py --schema path/to/schema.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

SCHEMA_URL = "https://schema.tauri.app/config/2"
#: ★ 官方 schema 随仓库分发（169KB）—— 让校验**离线可控**，可进日常门禁。
#: 不这样做的话：唯一能发现配置错误的是 `cargo check`，而它需要 Rust + MSVC 工具链，
#: CI/沙箱里跑不了 ⇒ 这类错误会一路漏到出包才炸。
COMMITTED_SCHEMA = Path(__file__).resolve().parents[1] / "desktop" / "src-tauri" / "tauri.schema.json"
CONFIG = Path("desktop/src-tauri/tauri.conf.json")


def load_schema(cache: Path | None) -> dict:
    """优先用仓库内 schema；缺失时联网取（并写入 cache 供下次复用）。"""
    if COMMITTED_SCHEMA.is_file():
        return json.load(open(COMMITTED_SCHEMA, encoding="utf-8"))
    if cache and cache.is_file():
        return json.load(open(cache, encoding="utf-8"))
    with urllib.request.urlopen(SCHEMA_URL, timeout=30) as r:  # noqa: S310 固定 https 官方地址
        data = json.loads(r.read().decode("utf-8"))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        open(cache, "w", encoding="utf-8", newline="").write(json.dumps(data, ensure_ascii=False))
    return data


def _resolve(node: dict, root: dict) -> dict:
    """跟随 `$ref`（只支持 #/definitions/X 与 #/$defs/X）。"""
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 10:
        ref = node["$ref"]
        assert ref.startswith("#/"), f"不支持外部引用：{ref}"
        cur: object = root
        for part in ref[2:].split("/"):
            cur = cur[part]  # type: ignore[index]
        node = cur  # type: ignore[assignment]
        seen += 1
    return node


def validate(inst, sch, root: dict, path: str, errs: list[str]) -> None:
    sch = _resolve(sch, root)
    if not isinstance(sch, dict):
        return

    # ★★ allOf：所有分支都必须通过。
    #    ⚠️ 早期版本漏了这一个分支 ⇒ Tauri schema 里 `"某字段": {"allOf":[{"$ref":"#/definitions/X"}]}`
    #    这种包装全部被**静默跳过**，校验器永远报"通过"。
    #    **一个永远通过的校验器比没有校验器更危险**（会给出虚假的安全感）。
    if "allOf" in sch:
        for b in sch["allOf"]:
            validate(inst, b, root, path, errs)
        # allOf 之外还可能有自己的 properties/type，继续往下走做补充校验

    # 组合关键字：任一分支通过即算通过；全失败时给**最有信息量**的那条
    for key in ("anyOf", "oneOf"):
        if key in sch:
            branches = sch[key]
            sub_errs: list[list[str]] = []
            for b in branches:
                e: list[str] = []
                validate(inst, b, root, path, e)
                if not e:
                    return
                sub_errs.append(e)
            # 选错误数最少的分支（通常就是最接近的匹配）
            best = min(sub_errs, key=len)
            errs.extend(best if best else [f"{path}: 不匹配 {key} 的任一分支"])
            return

    if "type" in sch:
        want = sch["type"]
        wants = want if isinstance(want, list) else [want]
        actual = (
            "object" if isinstance(inst, dict)
            else "array" if isinstance(inst, list)
            else "boolean" if isinstance(inst, bool)
            else "integer" if isinstance(inst, int)
            else "number" if isinstance(inst, float)
            else "string" if isinstance(inst, str)
            else "null" if inst is None
            else "unknown"
        )
        if actual not in wants and not (actual == "integer" and "number" in wants):
            errs.append(f"{path}: 类型应为 {wants}，实际 {actual}")
            return

    if "enum" in sch and inst not in sch["enum"]:
        errs.append(f"{path}: 取值 {inst!r} 不在允许范围 {sch['enum']}")

    if isinstance(inst, dict):
        props = sch.get("properties") or {}
        for req in sch.get("required") or []:
            if req not in inst:
                hint = ""
                if props.get(req, {}).get("oneOf"):
                    names = []
                    for b in props[req]["oneOf"]:
                        bb = _resolve(b, root)
                        names += list((bb.get("properties") or {}).keys())
                    if names:
                        hint = f"（该项的可用键：{sorted(set(names))}）"
                errs.append(f"{path}: 缺少必填字段 {req!r}{hint}")
        if sch.get("additionalProperties") is False:
            unknown = [k for k in inst if k not in props]
            for k in unknown:
                errs.append(f"{path}: 未知字段 {k!r}（允许：{sorted(props)}）")
        for k, v in inst.items():
            if k in props:
                validate(v, props[k], root, f"{path}.{k}", errs)

    elif isinstance(inst, list) and "items" in sch:
        for i, v in enumerate(inst):
            validate(v, sch["items"], root, f"{path}[{i}]", errs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--schema", default=None, help="本地 schema 路径（默认用仓库内 tauri.schema.json）")
    ap.add_argument("--no-cache", action="store_true", help="忽略缓存，强制联网取最新 schema")
    a = ap.parse_args()

    cache = Path(a.schema) if a.schema else Path("desktop/src-tauri/target/tauri2.schema.json")
    root = load_schema(None if a.no_cache else cache)
    cfg = json.load(open(a.config, encoding="utf-8"))

    errs: list[str] = []
    validate(cfg, root, root, "$", errs)
    if errs:
        print(f"❌ {a.config} 有 {len(errs)} 处不符合 Tauri 2 schema：\n")
        for e in errs:
            print("  -", e)
        return 1
    print(f"✅ {a.config} 通过 Tauri 2 schema 校验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
