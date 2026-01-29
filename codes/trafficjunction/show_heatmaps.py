import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from traffic_junction_env import TrafficJunctionConfig, TrafficJunctionEnv
from models import ModelConfig, make_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Traffic Junction: Comm vector norms + brake locations heatmaps."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to a saved .pt checkpoint.")
    p.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="analysis_section_D")
    p.add_argument("--greedy", action="store_true", help="Use greedy actions (default: sample).")

    return p.parse_args()


def _load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model_name = ckpt.get("model_name", "commnet")
    mcfg_dict = ckpt["model_cfg"]
    env_cfg_dict = ckpt["env_cfg"]

    env_cfg = TrafficJunctionConfig(**env_cfg_dict)
    mcfg = ModelConfig(**mcfg_dict)

    model = make_model(model_name, mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, model_name, mcfg, env_cfg


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, model_name, mcfg, env_cfg = _load_checkpoint(args.ckpt, device)

    H, W = env_cfg.height, env_cfg.width
    comm_sum = np.zeros((H, W), dtype=np.float64)
    comm_cnt = np.zeros((H, W), dtype=np.float64)

    brake_cnt = np.zeros((H, W), dtype=np.float64)
    active_cnt = np.zeros((H, W), dtype=np.float64)

    # Optional per-source breakdown (source inferred from route_id // 3)
    comm_sum_dir = np.zeros((4, H, W), dtype=np.float64)
    comm_cnt_dir = np.zeros((4, H, W), dtype=np.float64)
    brake_cnt_dir = np.zeros((4, H, W), dtype=np.float64)
    active_cnt_dir = np.zeros((4, H, W), dtype=np.float64)

    for ep in range(args.episodes):
        env = TrafficJunctionEnv(env_cfg, seed=ep * 10007 + 7)
        obs = env.reset()

        state = model.init_state(1, env_cfg.nmax, device) if hasattr(model, "init_state") else None

        done = False
        while not done:
            # Snapshot positions BEFORE the step for attribution of comm + brakes.
            pos_before = [(-1, -1)] * env_cfg.nmax
            src_before = [-1] * env_cfg.nmax
            active_before = np.zeros((env_cfg.nmax,), dtype=np.int8)

            for i, car in enumerate(env.cars):
                if car.active:
                    active_before[i] = 1
                    pos_before[i] = car.pos
                    # route_id = source_id*3 + kind
                    src_before[i] = int(car.route_id // 3) if car.route_id >= 0 else -1

            obs_t = torch.tensor(obs["obs"], device=device).unsqueeze(0)
            mask_t = torch.tensor(obs["mask_active"], device=device).unsqueeze(0)

            if model_name == "discrete_comm":
                out = model(obs_t, mask_t, comm_mode="greedy" if args.greedy else "sample", state=state)

            else:
                out = model(obs_t, mask_t, state=state)

            logits = out["action_logits"]
            state = out.get("next_state", None)

            if args.greedy:
                act = logits.argmax(dim=-1)

            else:
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()

            act_np = act.squeeze(0).cpu().numpy().astype(np.int64) # [N]

            # Comm vector norms (only if model exposes them: CommNetContinuous does).
            comm_vec = out.get("comm_vec", None)

            if comm_vec is not None:
                comm_norm = torch.norm(comm_vec.squeeze(0), dim=-1).cpu().numpy() # [N]

            else:
                comm_norm = None

            for i in range(env_cfg.nmax):
                if active_before[i] == 0:
                    continue

                y, x = pos_before[i]

                if y < 0:
                    continue

                active_cnt[y, x] += 1.0
                sid = src_before[i]

                if sid >= 0:
                    active_cnt_dir[sid, y, x] += 1.0

                if comm_norm is not None:
                    comm_sum[y, x] += float(comm_norm[i])
                    comm_cnt[y, x] += 1.0

                    if sid >= 0:
                        comm_sum_dir[sid, y, x] += float(comm_norm[i])
                        comm_cnt_dir[sid, y, x] += 1.0

                if act_np[i] == 0: # brake
                    brake_cnt[y, x] += 1.0

                    if sid >= 0:
                        brake_cnt_dir[sid, y, x] += 1.0

            obs, r, done, info = env.step(act_np)

    comm_avg = np.divide(comm_sum, comm_cnt, out=np.zeros_like(comm_sum), where=(comm_cnt > 0))
    brake_prob = np.divide(brake_cnt, active_cnt, out=np.zeros_like(brake_cnt), where=(active_cnt > 0))

    os.makedirs(args.outdir, exist_ok=True)

    np.save(os.path.join(args.outdir, "comm_norm_avg.npy"), comm_avg)
    np.save(os.path.join(args.outdir, "brake_prob.npy"), brake_prob)

    plt.figure()
    plt.title("Average norm of communication vectors")
    plt.imshow(comm_avg, origin="upper")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "comm_norm_avg.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.title("Brake locations probability")
    plt.imshow(brake_prob, origin="upper")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "brake_prob.png"), dpi=200)
    plt.close()

    for sid, name in enumerate(["N", "S", "W", "E"]):
        cavg = np.divide(comm_sum_dir[sid], comm_cnt_dir[sid], out=np.zeros((H, W)), where=(comm_cnt_dir[sid] > 0))
        bprob = np.divide(brake_cnt_dir[sid], active_cnt_dir[sid], out=np.zeros((H, W)), where=(active_cnt_dir[sid] > 0))

        plt.figure()
        plt.title(f"Comm norm avg — source {name}")
        plt.imshow(cavg, origin="upper")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"comm_norm_avg_source_{name}.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.title(f"Brake prob — source {name}")
        plt.imshow(bprob, origin="upper")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"brake_prob_source_{name}.png"), dpi=200)
        plt.close()

    print(f"Saved analysis to: {args.outdir}")


if __name__ == "__main__":
    main()
