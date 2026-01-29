import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch

from traffic_junction_env import TrafficJunctionConfig, TrafficJunctionEnv
from utils import set_global_seeds, save_json, now_str
from models import ModelConfig, make_model
from trainer import TrainConfig, train_reinforce, evaluate_failure_rate


MODELS = [
    ("independent", "Independent"),
    ("fully_connected", "Fully-connected"),
    ("discrete_comm", "Discrete comm."),
    ("commnet", "CommNet"),
]

MODULES = [
    ("mlp", "MLP"),
    ("rnn", "RNN"),
    ("lstm", "LSTM"),
]


def mean_std(xs):
    xs = np.array(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def _run_one_job(job: dict) -> dict:
    """Train + eval for a single (model, module, seed). Runs in a separate process."""

    model_key = job["model_key"]
    module_key = job["module_key"]
    sd = int(job["seed"])
    obs_dim = int(job["obs_dim"])
    env_cfg_dict = job["env_cfg"]
    outdir = job["outdir"]
    run_id = job["run_id"]
    device = job["device"]
    n_comm_symbols = int(job["n_comm_symbols"])

    env_cfg = TrafficJunctionConfig(**env_cfg_dict)

    set_global_seeds(sd)

    mcfg = ModelConfig(
        obs_dim=obs_dim,
        hidden_dim=int(job["hidden_dim"]),
        n_actions=2,
        n_agents=env_cfg.nmax,
        module=module_key,
        k_steps=int(job["k_steps"]),
        use_skip=True,
        comm_dim=int(job["hidden_dim"]),
        n_comm_symbols=n_comm_symbols,
    )
    model = make_model(model_key, mcfg)

    tcfg = TrainConfig(
        batch_episodes=int(job["batch_episodes"]),
        lr=float(job["lr"]),
        rms_alpha=float(job["rms_alpha"]),
        rms_eps=float(job["rms_eps"]),
        grad_clip=float(job["grad_clip"]),
        alpha_baseline=float(job["alpha_baseline"]),
        entropy_coef=float(job["entropy_coef"]),
        epochs=int(job["epochs"]),
        updates_per_epoch=int(job["updates_per_epoch"]),
        device=device,
        seed=sd,
        use_curriculum=bool(job["use_curriculum"]),
        p_arrive_start=float(job["p_arrive_start"]),
        p_arrive_end=float(job["p_arrive_end"]),
    )

    history = train_reinforce(model, env_cfg, tcfg, n_comm_symbols=n_comm_symbols)

    # After training, env_cfg.p_arrive will be the last curriculum value (0.2 by default).
    fr, mr = evaluate_failure_rate(
        model,
        env_cfg,
        int(job["eval_episodes"]),
        torch.device(device),
        seed=sd + 9999,
        greedy=True,
        n_comm_symbols=n_comm_symbols,
    )

    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{run_id}_{model_key}_{module_key}_seed{sd}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": model_key,
            "model_cfg": mcfg.__dict__,
            "env_cfg": env_cfg.__dict__,
            "train_cfg": tcfg.__dict__,
        },
        ckpt_path,
    )

    save_json(
        os.path.join(outdir, f"{run_id}_{model_key}_{module_key}_seed{sd}_history.json"),
        {"history": history, "failure_rate_pct": fr * 100.0, "mean_return": mr, "ckpt": ckpt_path},
    )

    return {
        "model_key": model_key,
        "module_key": module_key,
        "seed": sd,
        "failure_rate_pct": fr * 100.0,
        "mean_return": mr,
    }


def main() -> None:
    p = argparse.ArgumentParser()

    # Environment
    p.add_argument("--visibility", type=int, default=1)   # 1->3x3, 2->5x5, 0->none

    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--p_arrive_start", type=float, default=0.05)
    p.add_argument("--p_arrive_end", type=float, default=0.20)

    # Training (paper defaults!)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--updates_per_epoch", type=int, default=100)
    p.add_argument("--batch_episodes", type=int, default=288)

    p.add_argument("--eval_episodes", type=int, default=500)
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="results_table2_left")
    p.add_argument("--jobs", type=int, default=0, help="Number of parallel worker processes (0=auto).")
    p.add_argument("--no_parallel", action="store_true", help="Disable multiprocessing (run sequentially).")

    # Filters
    p.add_argument("--models", type=str, default="", help="Comma-separated model keys to run. Empty = all.")
    p.add_argument("--modules", type=str, default="", help="Comma-separated module keys to run. Empty = all.")

    # Model
    p.add_argument("--hidden_dim", type=int, default=50)
    p.add_argument("--k_steps", type=int, default=2)
    p.add_argument("--n_comm_symbols", type=int, default=10)

    # Optim / loss
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--rms_alpha", type=float, default=0.97)
    p.add_argument("--rms_eps", type=float, default=1e-6)
    p.add_argument("--alpha_baseline", type=float, default=0.03)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=5.0)

    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    model_filter = [s.strip() for s in args.models.split(",") if s.strip()] if args.models else []
    module_filter = [s.strip() for s in args.modules.split(",") if s.strip()] if args.modules else []

    models_to_run = [m for m in MODELS if (not model_filter or m[0] in model_filter)]
    modules_to_run = [m for m in MODULES if (not module_filter or m[0] in module_filter)]

    run_id = now_str()
    os.makedirs(args.outdir, exist_ok=True)

    env_cfg = TrafficJunctionConfig(
        p_arrive=args.p_arrive_start,
        visibility=args.visibility,
    )

    env0 = TrafficJunctionEnv(env_cfg, seed=seeds[0])
    obs0 = env0.reset()
    obs_dim = obs0["obs"].shape[-1]

    table = {} 
    raw = {}

    jobs = []
    for model_key, model_name in models_to_run:
        table.setdefault(model_name, {})
        raw.setdefault(model_name, {})

        for module_key, module_name in modules_to_run:
            table[model_name].setdefault(module_name, None)
            raw[model_name].setdefault(module_name, {"fail_rates_pct": [], "mean_returns": []})

            for sd in seeds:
                jobs.append({
                    "model_key": model_key,
                    "model_name": model_name,
                    "module_key": module_key,
                    "module_name": module_name,
                    "seed": sd,
                    "obs_dim": obs_dim,
                    "env_cfg": env_cfg.__dict__,
                    "outdir": args.outdir,
                    "run_id": run_id,
                    "device": args.device,
                    "hidden_dim": args.hidden_dim,
                    "k_steps": args.k_steps,
                    "n_comm_symbols": args.n_comm_symbols,
                    "epochs": args.epochs,
                    "updates_per_epoch": args.updates_per_epoch,
                    "batch_episodes": args.batch_episodes,
                    "eval_episodes": args.eval_episodes,
                    "lr": args.lr,
                    "rms_alpha": args.rms_alpha,
                    "rms_eps": args.rms_eps,
                    "alpha_baseline": args.alpha_baseline,
                    "entropy_coef": args.entropy_coef,
                    "grad_clip": args.grad_clip,
                    "use_curriculum": (not args.no_curriculum),
                    "p_arrive_start": args.p_arrive_start,
                    "p_arrive_end": args.p_arrive_end,
                })

    # Parallelization

    if args.no_parallel:
        results = []
        for j in jobs:
            results.append({**_run_one_job(j), "model_name": j["model_name"], "module_name": j["module_name"]})
    else:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        cpu = os.cpu_count() or 4
        max_workers = args.jobs if args.jobs and args.jobs > 0 else min(4, cpu)

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_run_one_job, j): j for j in jobs}

            for fut in as_completed(futs):
                j = futs[fut]
                outj = fut.result()
                outj["model_name"] = j["model_name"]
                outj["module_name"] = j["module_name"]
                results.append(outj)

    for r in results:
        mn = r["model_name"]
        un = r["module_name"]
        raw[mn][un]["fail_rates_pct"].append(float(r["failure_rate_pct"]))
        raw[mn][un]["mean_returns"].append(float(r["mean_return"]))

    for model_key, model_name in models_to_run:
        for module_key, module_name in modules_to_run:
            fail_rates = raw[model_name][module_name]["fail_rates_pct"]
            m, s = mean_std(fail_rates)
            table[model_name][module_name] = {"mean_fail_pct": m, "std_fail_pct": s}

    out = {"run_id": run_id, "env_cfg": env_cfg.__dict__, "seeds": seeds, "table2_left": table, "raw": raw}
    save_json(os.path.join(args.outdir, f"{run_id}_table2_left.json"), out)

    # Show
    print("\nTable 2 (left) - Failure rate (%) mean +/- std")
    header = ["Model"] + [mname for _, mname in modules_to_run]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))
    for model_key, model_name in models_to_run:
        row = [model_name]
        for _, module_name in modules_to_run:
            cell = table[model_name][module_name]
            row.append(f"{cell['mean_fail_pct']:.1f} +/- {cell['std_fail_pct']:.1f}")
        print(" | ".join(row))


if __name__ == "__main__":
    main()
