from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np

Coord = Tuple[int, int]  # (y, x) 0-indexed


@dataclass
class Car:
    active: bool = False
    route_id: int = -1
    route_kind: int = 1  # 0=left, 1=straight, 2=right (assigned at spawn)
    route_pos: int = 0
    tau: int = 0
    pos: Coord = (0, 0)


@dataclass
class TrafficJunctionConfig:
    height: int = 14
    width: int = 14
    max_steps: int = 40
    nmax: int = 10  

    p_arrive: float = 0.30

    r_collision: float = -10.0
    r_wait: float = -0.01

    # Observation
    visibility: int = 1  # local view radius (1 -> 3x3)
    include_route_channels: bool = False


class TrafficJunctionEnv:
    """Traffic Junction environment

    - Grid: 14x14
    - Up to 10 cars active simultaneously
    - Each car follows a pre-defined route; the policy chooses speed:
        0 = brake, 1 = gas
    - Collision if >=2 cars occupy same cell after movement
    - Reward each step:
        r = (#collision cells)*(-10) + sum_i (tau_i * -0.01)
    """

    ACTION_BRAKE = 0
    ACTION_GAS = 1

    def __init__(self, cfg: TrafficJunctionConfig, seed = None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.cars: List[Car] = [Car(active=False, pos=(0, 0)) for _ in range(cfg.nmax)]
        self.step_count = 0
        self.build_layout_and_routes()

    def build_layout_and_routes(self):
        H, W = self.cfg.height, self.cfg.width
        assert H == 14 and W == 14, "Default is 14x14"
        cy, cx = H // 2, W // 2

        # Croix largeur 2 (keep-right lanes)
        # vertical lanes: columns cx-1 and cx
        v_left = cx - 1
        v_right = cx
        # horizontal lanes: rows cy-1 and cy
        h_up = cy - 1
        h_down = cy

        # Road cells for rendering + observation
        self.road_cells = set()
        for y in range(H):
            self.road_cells.add((y, v_left))
            self.road_cells.add((y, v_right))
        for x in range(W):
            self.road_cells.add((h_up, x))
            self.road_cells.add((h_down, x))

        # 4 sources (one per incoming lane, keep-right)
        # North incoming goes DOWN on v_left
        # South incoming goes UP on v_right
        # West incoming goes RIGHT on h_down
        # East incoming goes LEFT on h_up
        self.sources = [
            (0, v_left), # 0: North
            (H - 1, v_right), # 1: South
            (h_down, 0), # 2: West
            (h_up, W - 1), # 3: East
        ]

        """
        # Build helper straight segments
        def col_down(x):  # y increasing
            return [(y, x) for y in range(0, H)]
        
        def col_up(x):    # y decreasing
            return [(y, x) for y in range(H - 1, -1, -1)]
        
        def row_right(y): # x increasing
            return [(y, x) for x in range(0, W)]
        
        def row_left(y):  # x decreasing
            return [(y, x) for x in range(W - 1, -1, -1)]
        """

        routes = []

        # From North (id 0): driving DOWN on v_left
        rN_straight = [(y, v_left) for y in range(0, H)]
        rN_left = [(y, v_left) for y in range(0, h_down + 1)] + [(h_down, x) for x in range(v_left + 1, W)]
        rN_right = [(y, v_left) for y in range(0, h_up + 1)] + [(h_up, x) for x in range(v_left - 1, -1, -1)]

        routes += [rN_left, rN_straight, rN_right]

        # From South (id 1): driving UP on v_right
        rS_straight = [(y, v_right) for y in range(H - 1, -1, -1)]
        rS_left = [(y, v_right) for y in range(H - 1, h_up - 1, -1)] + [(h_up, x) for x in range(v_right - 1, -1, -1)]
        rS_right = [(y, v_right) for y in range(H - 1, h_down - 1, -1)] + [(h_down, x) for x in range(v_right + 1, W)]

        routes += [rS_left, rS_straight, rS_right]

        # From West (id 2): driving RIGHT on h_down
        rW_straight = [(h_down, x) for x in range(0, W)]
        rW_left = [(h_down, x) for x in range(0, v_right + 1)] + [(y, v_right) for y in range(h_down - 1, -1, -1)]
        rW_right = [(h_down, x) for x in range(0, v_left + 1)] + [(y, v_left) for y in range(h_down + 1, H)]

        routes += [rW_left, rW_straight, rW_right]

        # From East (id 3): driving LEFT on h_up
        rE_straight = [(h_up, x) for x in range(W - 1, -1, -1)]
        rE_left = [(h_up, x) for x in range(W - 1, v_left - 1, -1)] + [(y, v_left) for y in range(h_up + 1, H)]
        rE_right = [(h_up, x) for x in range(W - 1, v_right - 1, -1)] + [(y, v_right) for y in range(h_up - 1, -1, -1)]

        routes += [rE_left, rE_straight, rE_right]
    
        self.routes = routes

        # Map each source to exactly 3 routes: [left, straight, right]
        self.source_to_routes = {
            0: [0, 1, 2],     # North: left/straight/right
            1: [3, 4, 5],     # South
            2: [6, 7, 8],     # West
            3: [9, 10, 11],   # East
        }


    def reset(self):
        self.step_count = 0

        for c in self.cars:
            c.active = False
            c.route_id = -1
            c.route_kind = 1
            c.route_pos = 0
            c.tau = 0
            c.pos = (0, 0)

        return self.get_obs()

    def step(self, actions: np.ndarray):
        assert actions.shape == (self.cfg.nmax,)
        self.step_count += 1

        # Spawn cars for this time step
        # We only spawn if the entry cell is free to avoid "free" collisions
        self.maybe_spawn()

        # desired positions
        desired: List[Coord] = []
        for i, car in enumerate(self.cars):
            if not car.active:
                desired.append(car.pos)
                continue

            a = int(actions[i])

            if a == self.ACTION_GAS:
                nxt = min(car.route_pos + 1, len(self.routes[car.route_id]) - 1)
                desired.append(self.routes[car.route_id][nxt])

            else:
                desired.append(car.pos)

        for i, car in enumerate(self.cars):
            if not car.active:
                continue

            if int(actions[i]) == self.ACTION_GAS:
                car.route_pos = min(car.route_pos + 1, len(self.routes[car.route_id]) - 1)

            car.pos = desired[i]

        # collisions (cells with >=2 active cars)
        occ = {}
        for car in self.cars:
            if car.active:
                occ[car.pos] = occ.get(car.pos, 0) + 1

        collision_cells = sum(1 for cnt in occ.values() if cnt >= 2)

        # deactivate cars that finished route
        passed = 0
        for car in self.cars:
            if car.active and car.pos == self.routes[car.route_id][-1]:
                # print(f"[PASS] car end reached | route_id={car.route_id} | pos={car.pos}")
                passed += 1
                car.active = False
                car.route_id = -1
                car.route_pos = 0
                car.tau = 0
                car.pos = (0, 0)

        # reward: collision penalty + sum_i (tau_i * r_wait)
        wait_pen = 0.0
        for car in self.cars:
            if car.active:
                wait_pen += car.tau * self.cfg.r_wait

        reward = collision_cells * self.cfg.r_collision + wait_pen

        # update tau AFTER reward
        for car in self.cars:
            if car.active:
                car.tau += 1

        done = self.step_count >= self.cfg.max_steps
        info = {"collision_cells": collision_cells, "passed": passed, "active": self.num_active, "t": self.step_count}

        return self.get_obs(), float(reward), bool(done), info

    def maybe_spawn(self):
        # Occupied cells (active cars) to prevent spawning into an occupied entry cell.
        occupied = {c.pos for c in self.cars if c.active}

        for sid, src in enumerate(self.sources):
            if self.num_active >= self.cfg.nmax:
                break
            if self.rng.random() >= self.cfg.p_arrive:
                continue

            # a car can only enter the grid if the entry cell is free.
            if src in occupied:
                continue

            inactive = [i for i, c in enumerate(self.cars) if not c.active]
            if not inactive:
                break
            car_idx = self.rng.choice(inactive)
            route_id = self.rng.choice(self.source_to_routes[sid])

            car = self.cars[car_idx]
            car.active = True
            car.route_id = route_id
            car.route_kind = int(route_id % 3) # 0=left,1=straight,2=right
            car.route_pos = 0
            car.tau = 0
            car.pos = self.routes[route_id][0] # source cell
            occupied.add(car.pos)

    def num_active(self):
        return sum(1 for c in self.cars if c.active)

    def get_obs(self):
        """
        Each observed car is described by 3 one-hot vectors:
          n: car identity (size Nmax)
          l: location on the 14x14 grid (size 196)
          r: route type (left/straight/right) (size 3)

        Each agent observes a (2v+1)x(2v+1) local neighborhood (v=1 -> 3x3).
        For each cell in the window, we place a multi-hot vector encoding (n,l,r) for the car(s) occupying that cell. Empty cells are all-zeros.
        """
        v = self.cfg.visibility
        win = 2 * v + 1
        n_dim = self.cfg.nmax
        l_dim = self.cfg.height * self.cfg.width
        r_dim = 3
        cell_dim = n_dim + l_dim + r_dim
        D = win * win * cell_dim

        obs = np.zeros((self.cfg.nmax, D), dtype=np.float32)
        mask = np.zeros((self.cfg.nmax,), dtype=np.float32)

        # position -> list of (car_index, route_kind)
        pos_to_cars: Dict[Coord, List[Tuple[int, int]]] = {}

        for idx, c in enumerate(self.cars):
            if not c.active:
                continue

            pos_to_cars.setdefault(c.pos, []).append((idx, int(c.route_kind)))

        for i, car in enumerate(self.cars):
            if not car.active:
                continue

            mask[i] = 1.0
            y0, x0 = car.pos
            out = []

            for dy in range(-v, v + 1):
                for dx in range(-v, v + 1):
                    y, x = y0 + dy, x0 + dx
                    cell = np.zeros((cell_dim,), dtype=np.float32)

                    if 0 <= y < self.cfg.height and 0 <= x < self.cfg.width:
                        here = (y, x)

                        if here in pos_to_cars:
                            loc_idx = y * self.cfg.width + x
                            cell[n_dim + loc_idx] = 1.0

                            for (car_j, rk) in pos_to_cars[here]:
                                cell[car_j] = 1.0
                                cell[n_dim + l_dim + rk] = 1.0

                    out.append(cell)

            obs[i] = np.concatenate(out, axis=0)

        return {"obs": obs, "mask_active": mask}

    def obs_dim(self):
        v = self.cfg.visibility
        win = 2 * v + 1
        n_dim = self.cfg.nmax
        l_dim = self.cfg.height * self.cfg.width
        r_dim = 3

        return win * win * (n_dim + l_dim + r_dim)
