"""CLI entry point. Run `safety-circuits --help` after `pip install -e .`."""

from __future__ import annotations

import argparse

from .ablation import HeadRef, evaluate_ablation
from .analysis import aggregate_pairs, head_heatmap, plot_heatmap
from .config import MODELS, ARTIFACTS_DIR, RESULTS_DIR
from .data import build_matched_pairs, load_advbench, load_hh_harmless, load_wikitext2
from .models import load_model
from .patching import patch_each_head


def cmd_run_mvp(args: argparse.Namespace) -> None:
    spec = MODELS[args.model]
    loaded = load_model(spec, device=args.device, dtype=args.dtype)

    harm = load_advbench(limit=args.n_pairs * 4)
    safe = load_hh_harmless(limit=args.n_pairs * 4)
    pairs = build_matched_pairs(harm, safe, n_pairs=args.n_pairs)

    print(f"Loaded {len(pairs)} pairs on {spec.key} "
          f"({loaded.n_layers} layers × {loaded.n_heads} heads).")

    per_pair = []
    for h, s in pairs:
        per_pair.append(patch_each_head(loaded, h.text, s.text))

    agg = aggregate_pairs(per_pair)
    RESULTS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    agg.to_csv(RESULTS_DIR / f"{spec.key}_patch_z.csv", index=False)

    grid = head_heatmap(agg, n_layers=loaded.n_layers, n_heads=loaded.n_heads)
    plot_heatmap(grid, title=f"{spec.key}: per-head |Δrefusal|",
                 save_to=str(RESULTS_DIR / f"{spec.key}_heatmap.png"))

    top = agg[agg["component"] == "z"].head(args.top_k)
    heads = [HeadRef(int(r.layer), int(r.head)) for r in top.itertuples()]
    print(f"Top-{args.top_k} candidate heads: {[(h.layer, h.head) for h in heads]}")

    eval_prompts = [p.text for p, _ in pairs[: args.n_pairs]]
    ppl_texts = load_wikitext2(limit=args.ppl_texts) if args.ppl_texts else None
    report = evaluate_ablation(loaded, heads, eval_prompts, mode="zero", perplexity_texts=ppl_texts)
    print(f"Refusal rate — clean: {report.refusal_rate_clean:.2%}  "
          f"ablated: {report.refusal_rate_ablated:.2%}")
    if report.perplexity_clean is not None:
        print(f"Perplexity   — clean: {report.perplexity_clean:.3f}  "
              f"ablated: {report.perplexity_ablated:.3f}  (Δ {report.perplexity_pct_change:+.2f}%)")


def main() -> None:
    p = argparse.ArgumentParser("safety-circuits")
    sub = p.add_subparsers(dest="cmd", required=True)

    mvp = sub.add_parser("run-mvp", help="end-to-end thin slice")
    mvp.add_argument("--model", default="qwen2.5", choices=list(MODELS))
    mvp.add_argument("--device", default="auto")
    mvp.add_argument("--dtype", default="float32")
    mvp.add_argument("--n_pairs", type=int, default=8)
    mvp.add_argument("--top_k", type=int, default=10)
    mvp.add_argument("--ppl_texts", type=int, default=16,
                     help="WikiText-2 snippets for the perplexity control (0 to skip)")
    mvp.set_defaults(func=cmd_run_mvp)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
