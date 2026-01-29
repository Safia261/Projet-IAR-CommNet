import argparse
import os
import torch

from traffic_junction_env import TrafficJunctionConfig, TrafficJunctionEnv
from models import ModelConfig, make_model
from trainer import TrainConfig, train_reinforce, evaluate_failure_rate
from utils import set_global_seeds, save_json, now_str


def parse_args():
    p = argparse.ArgumentParser(
        description="Train one controller on Traffic Junction with REINFORCE + baseline (CommNet paper Appendix A/C)."
    )

    # Model
    p.add_argument("--model", type=str, default="commnet",
                   choices=["independent", "fully_connected", "discrete_comm", "commnet"])
    p.add_argument("--module", type=str, default="mlp", choices=["mlp", "rnn", "lstm"])
    p.add_argument("--hidden_dim", type=int, default=50)
    p.add_argument("--k_steps", type=int, default=2)
    p.add_argument("--n_comm_symbols", type=int, default=10)

    # Env
    p.add_argument("--p_arrive", type=float, default=0.05,
                   help="Initial p_arrive (overridden by curriculum unless --no_curriculum).")
    p.add_argument("--visibility", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--nmax", type=int, default=10)
    p.add_argument("--r_collision", type=float, default=-10.0)
    p.add_argument("--r_wait", type=float, default=-0.01)

    # Train
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--updates_per_epoch", type=int, default=100)
    p.add_argument("--batch_episodes", type=int, default=288)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")

    # Optim
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--rms_alpha", type=float, default=0.97)
    p.add_argument("--rms_eps", type=float, default=1e-6)
    p.add_argument("--alpha_baseline", type=float, default=0.03,
                   help="Appendix A: alpha for baseline regression loss.")
    p.add_argument("--entropy_coef", type=float, default=0.0,
                   help="Not in the paper; keep 0.0 for faithful reproduction.")
    p.add_argument("--grad_clip", type=float, default=5.0)

    # Curriculum (Appendix C)
    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--p_arrive_start", type=float, default=0.05)
    p.add_argument("--p_arrive_end", type=float, default=0.20)

    # Eval / outputs
    p.add_argument("--eval_episodes", type=int, default=500)
    p.add_argument("--outdir", type=str, default="runs")

    return p.parse_args()


def main():
    args = parse_args()

    set_global_seeds(args.seed)
    device = torch.device(args.device)

    env_cfg = TrafficJunctionConfig(
        nmax=args.nmax,
        max_steps=args.max_steps,
        p_arrive=args.p_arrive,
        r_collision=args.r_collision,
        r_wait=args.r_wait,
        visibility=args.visibility,
    )

    env0 = TrafficJunctionEnv(env_cfg, seed=args.seed)
    obs0 = env0.reset()
    obs_dim = obs0["obs"].shape[-1]

    mcfg = ModelConfig(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        n_actions=2,
        n_agents=env_cfg.nmax,
        module=args.module,
        k_steps=args.k_steps,
        use_skip=True,
        comm_dim=args.hidden_dim,
        n_comm_symbols=args.n_comm_symbols,
    )

    model = make_model(args.model, mcfg)

    tcfg = TrainConfig(
        batch_episodes=args.batch_episodes,
        lr=args.lr,
        rms_alpha=args.rms_alpha,
        rms_eps=args.rms_eps,
        grad_clip=args.grad_clip,
        alpha_baseline=args.alpha_baseline,
        entropy_coef=args.entropy_coef,
        epochs=args.epochs,
        updates_per_epoch=args.updates_per_epoch,
        device=args.device,
        seed=args.seed,
        use_curriculum=(not args.no_curriculum),
        p_arrive_start=args.p_arrive_start,
        p_arrive_end=args.p_arrive_end,
    )

    run_id = now_str()
    outdir = os.path.join(args.outdir, run_id)
    os.makedirs(outdir, exist_ok=True)

    history = train_reinforce(model, env_cfg, tcfg, n_comm_symbols=args.n_comm_symbols)

    fr, mr = evaluate_failure_rate(
        model,
        env_cfg,
        episodes=args.eval_episodes,
        device=device,
        seed=args.seed + 9999,
        greedy=True,
        n_comm_symbols=args.n_comm_symbols,
    )

    ckpt_path = os.path.join(outdir, f"model_{args.model}_{args.module}_seed{args.seed}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "model_name": args.model,
        "model_cfg": mcfg.__dict__,
        "env_cfg": env_cfg.__dict__,
        "train_cfg": tcfg.__dict__,
        "failure_rate": fr,
        "mean_return": mr,
    }, ckpt_path)

    save_json(os.path.join(outdir, "summary.json"), {
        "run_id": run_id,
        "model": args.model,
        "module": args.module,
        "seed": args.seed,
        "env_cfg": env_cfg.__dict__,
        "model_cfg": mcfg.__dict__,
        "train_cfg": tcfg.__dict__,
        "failure_rate": fr,
        "failure_rate_pct": fr * 100.0,
        "mean_return": mr,
        "ckpt": ckpt_path,
    })

    print(f"\nDone. failure_rate={fr*100:.2f}% mean_return={mr:.3f}")
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
