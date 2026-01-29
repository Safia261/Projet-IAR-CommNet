import torch
import time
import random
import numpy as np
import tkinter as tk

class CombatTask:
    """
    Environnement pour le jeu de combat task multi-agent.
    2 équipes chacunes de 5 agnts: l'une de bots, l'autre controlée par un modèle.
    ...
    """

    def __init__(
            self,
            grid_size = 15,
            max_steps = 40,
            n_agents_per_team = 5, # = m
            visual_range =  1, # distance max -> donc carré 3x3
            fire_range = 1, # distance max -> donc carré 3x3
            initial_health = 3,
            cooldown_time = 1,
            device = "cpu",
            render_mode = True):
        
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.visual_range = visual_range
        self.fire_range = fire_range
        self.initial_health = initial_health
        self.cooldown_time = cooldown_time

        self.device = device
        self.render_mode = render_mode

        # Equipes 
        self.n_agents_per_team = n_agents_per_team
        self.n_teams = 2
        self.n_agents_total = self.n_agents_per_team * self.n_teams

        # Actions: 4 actions de déplacement, m attaques, 1 action de rester sur place
        self.n_move_actions = 4
        self.n_attack_actions = self.n_agents_per_team
        self.n_actions = self.n_move_actions + self.n_attack_actions + 1 # car action de rester sur place

        # Dimensions des one-hot
        self.id_dim = self.n_agents_total 
        self.team_dim = 2 
        self.loc_dim = grid_size * grid_size 
        self.hp_dim = initial_health + 1 
        self.cooldown_dim = 2
        self.feature_dim = ( self.id_dim + self.team_dim + self.loc_dim + self.hp_dim + self.cooldown_dim )
        self.obs_dim = self.feature_dim * 2 # self + neighbors

        if render_mode: 
            self.window = tk.Tk() 
            self.window.title("Combat Task") 
            self.canvas = tk.Canvas(self.window, width=520, height=520, bg="white") 
            self.canvas.pack()

        self.reset()


    ############################################
    # Foncitons utiles sur les agents et les équipes
    ############################################

    def _create_team(self, team_id):
        """
        Crée une équipe d'agents avec des positions initiales dans une zone spécifique de la grille (carré 5x5).
        """
        agents = []

        # choisit un centre aléatoire (disponible)
        cx = random.randint(0, self.grid_size - 1)
        cy = random.randint(0, self.grid_size - 1)

        # limites de la zone 5x5
        min_x = max(0, cx - 2)
        max_x = min(self.grid_size - 1, cx + 2)
        min_y = max(0, cy - 2)
        max_y = min(self.grid_size - 1, cy + 2)

        positions = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        random.shuffle(positions)

        for i in range(self.n_agents_per_team):
            a = {
                "id": i,
                "team": team_id,
                # "position": (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)),
                "position": positions[i],
                "health": self.initial_health,
                "cooldown": 0,
                "alive": True
            }
            agents.append(a)
        
        return agents
        
    def get_all_alive_agents(self):
        """
        Renvoie la liste de tous les agents vivants.
        """
        return [agent for agent in self.blue_team + self.red_team if agent["alive"]]
    
    def global_agent_index(self, agent):
        if agent["team"] == 0:
            return agent["id"]
        else:
            return self.n_agents_per_team + agent["id"]
    
    def all_agents(self):
        return self.blue_team + self.red_team

    ############################################
    # 
    ############################################
    def reset(self):
        """
        Réinitialise l'environnement à l'état initial.
        Place les agents sur la grille et initialise les variables d'état.
        """
        self.t = 0
        # Equipe bleue = équipe de bots (team_id = 0)
        self.blue_team = self._create_team(team_id = 0)
        # Equipe rouge = équipe contrôlée par le modèle (team_id = 1)
        self.red_team = self._create_team(team_id = 1)

        # vérifier si y a des collisions de positions:
        # Empêcher les collisions entre équipes au spawn
        blue_positions = {tuple(a["position"]) for a in self.blue_team}
        red_positions = {tuple(a["position"]) for a in self.red_team}

        # Si collision → respawn la red team
        while len(blue_positions.intersection(red_positions)) > 0:
            self.red_team = self._create_team(1)
            red_positions = {tuple(a["position"]) for a in self.red_team}


        if self.render_mode:
            self.render()
        return self.get_obs()
    

    def step(self, red_actions):
        """
        Exécute une étape dans l'environnement en fonction des actions des agents.
        Actions possibles : 'move_up', 'move_down', 'move_left', 'move_right', 'fire', 'stay'
        """
        self.t += 1
        done = False
        
        blue_actions = self.bot_policy()

        # Apply move actions for both teams (only if actions are move actions)
        self.apply_moves(self.red_team, red_actions)
        self.apply_moves(self.blue_team, blue_actions)

        # Apply attack actions for both teams (only if actions are attack actions)
        self.apply_attack(self.red_team, red_actions, target_team = self.blue_team)
        self.apply_attack(self.blue_team, blue_actions, target_team = self.red_team)

        # Decrease cooldown time for all agents having cooldown > 0
        for agent in self.get_all_alive_agents():
            if agent["cooldown"] > 0:
                agent["cooldown"] -= 1
        
        done, result = self.check_end_combat()
        reward = self.compute_reward(result)

        if self.render_mode:
            self.render()

        return self.get_obs(), reward, done, {"result": result}


    ############################################
    # 
    ############################################
    def apply_moves(self, team, actions):
        """
        Applique les actions de déplacements pour une équipe donnée.
        Actions de déplacements: 0=up, 1=down, 2=left, 3=right.
        """
        for agent, action in zip(team, actions):
            if not agent["alive"]:
                continue
                
            if action < self.n_move_actions:
                dx, dy = 0, 0
                if action == 0: # up
                    dx, dy = -1, 0
                elif action == 1: # down
                    dx, dy = 1, 0
                elif action == 2: # left
                    dx, dy = 0, -1
                elif action == 3: # right
                    dx, dy = 0, 1

                x, y = agent["position"]
                nx = min(max(0, x + dx), self.grid_size - 1)
                ny = min(max(0, y + dy), self.grid_size - 1)
                agent["position"] = (nx, ny)
    

    def apply_attack(self, team, actions, target_team):
        """
        Appliquer les actions d'attaque pour une équipe donnée contre une autre équipe.
        Actions d'attaque: m attaques.
        """
        for agent, action in zip(team, actions):
            if not agent["alive"]:
                continue

            if agent["cooldown"] > 0: # pas d'attaque possible pour cet agent
                # agent["cooldown"] -= 1
                continue

            if self.n_move_actions <= action < self.n_move_actions + self.n_attack_actions:
                target_id = action - self.n_move_actions # m attaques qui ciblent chacune un ennemi

                if target_id < len(target_team):
                    target_agent = target_team[target_id]
                    if target_agent["alive"]:
                        ax, ay = agent["position"]
                        tx, ty = target_agent["position"]
                        if abs(ax - tx) <= self.fire_range and abs(ay - ty) <= self.fire_range:
                            target_agent["health"] -= 1
                            if target_agent["health"] <= 0:
                                target_agent["alive"] = False
                            agent["cooldown"] = self.cooldown_time
    
    def check_end_combat(self):
        """
        Vérifie si le combat est terminé: 
        - tous les agents d'une équipe sont morts -> une équipe gagnante
        OU
        - le nombre maximum de steps est atteint: égalisation.
        """
        blue_alive = any(agent["alive"] for agent in self.blue_team)
        red_alive = any(agent["alive"] for agent in self.red_team)

        if not blue_alive and red_alive: # bots morts, équipe controllée par le modèle gagne
            return True, "win"
        if not red_alive and blue_alive: # équipe controllée par le modèle morte, bots gagnent
            return True, "lose"
        if not red_alive and not blue_alive: # les deux équipes mortes -> cas qui devrait être rare?
            return True, "draw" 
        if self.t >= self.max_steps:
            return True, "draw"
        return False, None
    

    def compute_reward(self, result):
        """
        Calcule la récompense pour l'équipe controlée par le modèle.
        """
        if result is None:
            return 0.0
        if result in ["loss", "draw"]:
            r = -1.0
        else:
            r = 0.0
        
        enemy_health_pts = sum(agent["health"] for agent in self.blue_team if agent["alive"])
        return r + (enemy_health_pts * -0.1)


    ############################################
    # Bot policy
    ############################################
    def in_visual_range(self, a, b):
        (x1, y1), (x2, y2) = a["position"], b["position"]
        return abs(x1 - x2) <= self.visual_range and abs(y1 - y2) <= self.visual_range

    def in_firing_range(self, a, b):
        (x1, y1), (x2, y2) = a["position"], b["position"]
        return abs(x1 - x2) <= self.fire_range and abs(y1 - y2) <= self.fire_range

    def closest_agent(self, a, b):
        (x1, y1), (x2, y2) = a["position"], b["position"]
        return max(abs(x1 - x2), abs(y1 - y2))

    def bot_policy(self):
        actions = []
        enemies = [agent for agent in self.red_team if agent["alive"]]

        if len(enemies) == 0: # si tous les ennemis sont déjà morts
            return [self.n_actions -1] * len(self.blue_team)
        
        # vision paratagée des bots -> partagent leur vision entre eux
        visible = set()
        for e in enemies:
            for bot in self.blue_team:
                if bot["alive"] and self.in_visual_range(bot, e):
                    # visible.add(e)
                    visible.add(self.global_agent_index(e))
                    break
        visible = list(visible)

        for bot in self.blue_team:
            if not bot["alive"]:
                actions.append(self.n_actions -1)
                continue

            # Attaque d'un ennemi proche (dans firing range)
            e_in_range = [e for e in enemies if self.in_firing_range(bot, e)] # 1 ennemi peut être attaqué par 2 bots en même temps ??
            if len(e_in_range) > 0:
                target = min(e_in_range, key=lambda e:self.closest_agent(bot, e))
                actions.append(self.n_move_actions + target["id"])
                continue

            # sinon avancer vers l'ennemi visible le plus proche
            if len(visible) > 0:
                target_id = min(visible, key=lambda g_id: self.closest_agent(bot, self.all_agents()[g_id]))
                target = self.all_agents()[target_id]
                bx, by = bot["position"]
                tx, ty = target["position"]
                dx, dy = tx - bx, ty - by
                if abs(dx) >= abs(dy): # déplacement vertical
                    if dx < 0: actions.append(0) # ennemi en haut
                    elif dx > 0: actions.append(1) # ennemi en bas
                    else:
                        if dy < 0: actions.append(2) # ennemi à gauche
                        elif dy > 0: actions.append(3) # ennemi à droite
                        else: actions.append(self.n_actions - 1) # sinon (si aucun ennemi proche), reste sur place
                else: # déplacement horizontal
                    if dy < 0: actions.append(2) # ennemi à gauche
                    elif dy > 0: actions.append(3) # ennemi à droite
                    else:
                        if dx < 0: actions.append(0) # ennemi en haut
                        elif dx > 0: actions.append(1) # ennemi en bas
                        else: actions.append(self.n_actions - 1)
                continue

            # sinon aucune opération
            #actions.append(self.n_actions - 1)
            # sinon déplacement aléatoire
            actions.append(random.randint(0, self.n_move_actions - 1))
            #if random.random() < 0.7:
            #    # 70% du temps : mouvement aléatoire
            #    actions.append(random.randint(0, self.n_move_actions - 1))
            #else:
            #    # 30% du temps : rester sur place
            #    actions.append(self.n_actions - 1)

        return actions

            
    
    ############################################
    # Encodage des observations des agents (red team) et de leurs voisins
    ############################################
    def one_hot_binary_vector(self, index, size):
        v = np.zeros(size, dtype=np.float32)
        v[index] = 1.0 # ou int(1) ???
        return v

    def encode_agent_features(self, agent):
        g_id = self.global_agent_index(agent)
        id_vec = self.one_hot_binary_vector(g_id, self.id_dim)

        team_vec = self.one_hot_binary_vector(agent["team"], self.team_dim)

        x, y = agent["position"]
        loc_index = x * self.grid_size + y
        loc_vec = self.one_hot_binary_vector(loc_index, self.loc_dim)

        hp_vec = self.one_hot_binary_vector(agent["health"], self.hp_dim)
        cd_vec = self.one_hot_binary_vector(1 if agent["cooldown"] > 0 else 0, self.cooldown_dim)

        return np.concatenate([id_vec, team_vec, loc_vec, hp_vec, cd_vec])

    def encode_neighbours(self, agent):
        ax, ay = agent["position"]
        features_sum = np.zeros(self.feature_dim, dtype=np.float32) # ou np.int ??
        
        for other in self.get_all_alive_agents():
            if other is agent:
                continue
            if self.in_visual_range(agent, other):
                features_sum += self.encode_agent_features(other)
        return features_sum


    def get_obs(self):
        """
        Retourne les observations pour chaque agent.
        Chaque agent voit les positions relatives et la santé des agents ennemis dans son champ de vision.
        """
        obs = []
        for agent in self.red_team:
            if not agent["alive"]:
                self_feat = np.zeros(self.feature_dim)
                neigh_feat = np.zeros(self.feature_dim)
            else:
                self_feat = self.encode_agent_features(agent)
                neigh_feat = self.encode_neighbours(agent)
            obs.append(np.concatenate([self_feat, neigh_feat]))
        # return obs
        return np.array(obs, dtype=np.float32)
    

    ############################################
    # Rendu visuel avec Tkinter
    ############################################
    def render(self):
        if not self.render_mode:
            return

        self.canvas.delete("all")

        cell_size = 520 // self.grid_size

        # Grille
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray90")

        # agents
        def draw_agent(agent, color):
            if not agent["alive"]:
                return
            x, y = agent["position"]
            cx = y * cell_size + cell_size // 2
            cy = x * cell_size + cell_size // 2
            r = cell_size * 0.35

            # Cercle
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=color, outline="black", width=2
            )

            # ID au centre
            self.canvas.create_text(
                cx, cy,
                text=str(agent["id"] + 1),
                fill="white",
                font=("Arial", int(cell_size * 0.35), "bold")
            )

        # Blue team (bots)
        for agent in self.blue_team:
            draw_agent(agent, "blue")

        # Red team (model)
        for agent in self.red_team:
            draw_agent(agent, "red")

        self.window.update()
    

def main():
    env = CombatTask(render_mode=True)

    obs = env.reset()

    done = False
    while not done:
        # actions = [env.n_actions - 1] * env.n_agents_per_team  # no-op
        actions = [random.randint(0, env.n_actions - 1) for _ in range(env.n_agents_per_team)]
        obs, reward, done, info = env.step(actions)
        # print("Actions : ", actions)
        time.sleep(0.2)
    
    print("Reward:", reward, " Success ? " , info)

if __name__ == "__main__":
    main()