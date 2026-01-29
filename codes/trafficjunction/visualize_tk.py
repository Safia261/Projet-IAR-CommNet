import argparse
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, Tuple, List

import numpy as np
import torch

from traffic_junction_env import TrafficJunctionEnv, TrafficJunctionConfig
from models import ModelConfig, make_model

CELL = 32

class TrafficViewer(tk.Tk):
    def __init__(self, env: TrafficJunctionEnv, model, device: torch.device):
        super().__init__()
        self.title("Traffic Junction (2x2, keep-right)")

        self.env = env
        self.model = model
        self.device = device

        self.canvas = tk.Canvas(self,
            width=env.cfg.width * CELL,
            height=env.cfg.height * CELL,
            bg="white",
        )
        self.canvas.grid(row=0, column=0, columnspan=4, padx=6, pady=6)

        self.btn_reset = ttk.Button(self, text="Reset", command=self.reset_env)
        self.btn_step = ttk.Button(self, text="Step", command=self.step_once)
        self.btn_play = ttk.Button(self, text="Play episode", command=self.play_episode)
        self.lbl = ttk.Label(self, text="")

        self.btn_reset.grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        self.btn_step.grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        self.btn_play.grid(row=1, column=2, padx=6, pady=6, sticky="ew")
        self.lbl.grid(row=2, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="w")

        self.obs = None
        self.state = None
        self.t = 0
        self.collision_count = 0
        self.total_r = 0.0
        self.last_actions = np.zeros((self.env.cfg.nmax,), dtype=int)  # 0=brake, 1=gas
        self.just_finished: Dict[int, Tuple[int,int,int]] = {}  # idx -> (y,x,route_kind) for 1-frame display

        self.reset_env()

    def reset_env(self):
        self.obs = self.env.reset()
        self.t = 0
        self.collision_count = 0
        self.total_r = 0.0
        self.last_actions = np.zeros((self.env.cfg.nmax,), dtype=int)
        self.just_finished = {}

        if hasattr(self.model, "init_state"):
            self.state = self.model.init_state(1, self.env.cfg.nmax, self.device)

        else:
            self.state = None

        self.render(info={"collision_cells": 0})

    @torch.no_grad()
    def policy_action(self) -> np.ndarray:
        obs_t = torch.tensor(self.obs["obs"], device=self.device).unsqueeze(0)
        mask_t = torch.tensor(self.obs["mask_active"], device=self.device).unsqueeze(0)

        out = self.model(obs_t, mask_t, state=self.state)
        act_logits = out["action_logits"]
        act = act_logits.argmax(dim=-1)  # greedy
        self.state = out.get("next_state", None)

        return act.squeeze(0).cpu().numpy().astype(int)

    def step_once(self):
        # Snapshot before stepping so we can show cars on their destination cell
        # for exactly one frame (the env deactivates them immediately after reaching it).
        pre_active = [c.active for c in self.env.cars]
        pre_dest = []
        pre_kind = []
        for c in self.env.cars:
            if c.active and c.route_id >= 0:
                dest = self.env.routes[c.route_id][-1]

            else:
                dest = (0, 0)

            pre_dest.append(dest)
            pre_kind.append(c.route_kind)

        a = self.policy_action()
        self.last_actions = a.copy()

        obs, r, done, info = self.env.step(a)
        self.obs = obs

        # Cars that were active but became inactive this step: draw them once at dest.
        self.just_finished = {}
        for i, was_active in enumerate(pre_active):
            if was_active and (not self.env.cars[i].active):
                dy, dx = pre_dest[i]
                self.just_finished[i] = (dy, dx, pre_kind[i])

        self.total_r += float(r)
        self.t += 1
        self.collision_count += int(info.get("collision_cells", 0))

        self.render(info)

        if done:
            self.lbl.configure(
                text=f"done | t={self.t} | collisions_total={self.collision_count} | return={self.total_r:.2f}"
            )

    def play_episode(self):
        self.reset_env()
        for _ in range(self.env.cfg.max_steps):
            self.update()
            self.step_once()
            self.update()
            time.sleep(0.5)

            if self.t >= self.env.cfg.max_steps:
                break

    def render(self, info):
        self.canvas.delete("all")

        H, W = self.env.cfg.height, self.env.cfg.width
        cy, cx = H // 2, W // 2 # intersection center
        # Lanes
        v_left = cx - 1 # North->South 
        v_right = cx # South->North
        h_up = cy - 1 # East->West
        h_down = cy # West->East 
        row_EW = h_down
        row_WE = h_up
        col_SN = v_left
        col_NS = v_right

        # grid
        for y in range(H):
            for x in range(W):
                self.rect(y, x, fill="", outline="#ddd")

        # roads (two lanes)
        for x in range(W):
            self.rect(row_EW, x, fill="#e6e6e6", outline="#ddd")
            self.rect(row_WE, x, fill="#e6e6e6", outline="#ddd")

        for y in range(H):
            self.rect(y, col_SN, fill="#e6e6e6", outline="#ddd")
            self.rect(y, col_NS, fill="#e6e6e6", outline="#ddd")

        # Source = env.sources; Destinations = exits (end of each route).
        for (sy, sx) in self.env.sources:
            self.label(sy, sx, "S")

        dests = {route[-1] for route in self.env.routes}

        for (dy, dx) in dests:
            self.label(dy, dx, "D")

        # occupancy -> collision cells
        pos_to_ids: Dict[Tuple[int, int], List[int]] = {}
        for i, c in enumerate(self.env.cars):
            if not c.active:
                continue

            y, x = c.pos
            pos_to_ids.setdefault((y, x), []).append(i)

        collision_positions = {pos for pos, ids in pos_to_ids.items() if len(ids) >= 2}

        for (y, x) in collision_positions:
            self.rect(y, x, fill="#ffcccc", outline="#ff0000")

        # cars
        for i, c in enumerate(self.env.cars):
            if not c.active:
                continue

            y, x = c.pos
            pad = 4
            fill = self.car_color(c.route_kind)

            # outline priority: collision (red) > brake (blue) > default (black)
            is_colliding = (y, x) in collision_positions
            is_braking = (self.last_actions[i] == 0)

            if is_colliding:
                outline, width = "red", 4

            elif is_braking:
                outline, width = "blue", 3

            else:
                outline, width = "black", 2

            self.canvas.create_oval(
                x * CELL + pad,
                y * CELL + pad,
                x * CELL + CELL - pad,
                y * CELL + CELL - pad,
                fill=fill,
                outline=outline,
                width=width,
            )

            letter = "L" if c.route_kind == 0 else "S" if c.route_kind == 1 else "R"
            self.canvas.create_text(
                x * CELL + CELL / 2,
                y * CELL + CELL / 2,
                text=letter,
                fill="white",
                font=("Arial", 10, "bold"),
            )

            """
            if self.last_actions[i] == 0:  # brake
                self.rect(y, x, fill="#fff06c", outline="#ffd700")
            """

        # cars that just finished this step (env deactivates them immediately)
        for i, (y, x, kind) in getattr(self, 'just_finished', {}).items():
            pad = 4
            fill = self.car_color(kind)
            self.canvas.create_oval(
                x * CELL + pad,
                y * CELL + pad,
                x * CELL + CELL - pad,
                y * CELL + CELL - pad,
                fill=fill,
                outline='green',
                width=3,
            )
            letter = 'L' if kind == 0 else 'S' if kind == 1 else 'R'
            self.canvas.create_text(
                x * CELL + CELL / 2,
                y * CELL + CELL / 2,
                text=letter,
                fill='white',
                font=('Arial', 10, 'bold'),
            )

        self.lbl.configure(
            text=f"t={self.t} | collision_cells={info.get('collision_cells', 0)} | "
                 f"collisions_total={self.collision_count} | return={self.total_r:.2f}"
        )

    def rect(self, y, x, fill="", outline="#ddd"):
        self.canvas.create_rectangle(
            x * CELL,
            y * CELL,
            (x + 1) * CELL,
            (y + 1) * CELL,
            fill=fill,
            outline=outline,
        )

    def label(self, y, x, txt):
        self.canvas.create_text(
            x * CELL + 9,
            y * CELL + 9,
            text=txt,
            fill="black",
            font=("Arial", 10, "bold"),
        )

    def car_color(self, route: int):
        if route == 0:
            return "#2ca02c"  # left
        if route == 1:
            return "#1f77b4"  # straight
        return "#d62728"      # right


def load_model(ckpt_path, model_key, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_cfg"])
    model = make_model(model_key, mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    env_cfg = TrafficJunctionConfig(**ckpt["env_cfg"])

    return model, env_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model", type=str, required=True,
                   choices=["independent", "fully_connected", "discrete_comm", "commnet"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, env_cfg = load_model(args.ckpt, args.model, args.device)
    env = TrafficJunctionEnv(env_cfg, seed=args.seed)
    app = TrafficViewer(env, model, torch.device(args.device))
    app.mainloop()


if __name__ == "__main__":
    main()