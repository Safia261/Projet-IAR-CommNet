import torch
import time
import random
import tkinter as tk

# 2 actions possibles
BREAK = 0           # arrêt
GAS = 1             # avancer

class TrafficJunction:
    """
    Environnement pour le jeu Traffic Junction:

    - N de voitures / agents dynamique au cours de la simulation (max = 10)
    - grille 14x14, avec 4 routes (N, S, W, E) double-file se croisant 
    - probabilité p_arrive qu'une nouvelle voiture arrive sur l'une des routes
    - But: éviter toute collision (en communiquant)
    - communication en broadcast
    - champ de vision 3x3
    - 40 steps au max pour la simulation
    """

    def __init__(
            self, 
            grid_size = 14, 
            max_steps = 40,
            max_agents = 10,
            p_arrive =  0.3,
            device="cpu",
            render_mode = True):
        
        self.grid_size = grid_size
        self.center = grid_size // 2
        self.max_steps = max_steps
        self.device = device
        self.max_agents = max_agents
        self.p_arrive = p_arrive
        self.render_mode = render_mode
        self.collision_count = 0
        self.global_reward = 0.0

        if render_mode: 
            self.window = tk.Tk() 
            self.window.title("Traffic Junction") 
            self.canvas = tk.Canvas(self.window, width=520, height=520, bg="white") 
            self.canvas.pack()

        self.reset()


    def reset(self):
        """
        Initialise l'environnement. Aucune voiture initialement.
        """
        self.t = 0
        self.done = False

        self.num_agents = 0
        self.positions = torch.zeros((self.max_agents, 2), dtype=torch.long, device=self.device)
        self.directions = torch.zeros((self.max_agents, 2), dtype=torch.long, device=self.device)
        self.route_id = torch.full((self.max_agents,), -1, dtype=torch.long, device=self.device)
        self.has_turned = torch.zeros(self.max_agents, dtype=torch.bool, device=self.device)
        self.active = torch.zeros(self.max_agents, dtype=torch.bool, device=self.device)
        self.age = torch.zeros(self.max_agents, dtype=torch.long, device=self.device)
        self.collision_count = 0
        self.global_reward = 0.0

        return self.get_obs()


    def arriving_car(self):
        """
        Apparition d'une nouvelle voiture selon une probabilité d'apparition (p_arrive)
        """
        if self.num_agents >= self.max_agents:
            return

        if random.random() > self.p_arrive:
            return
        
        # 4 routes possibles et 2 voies (=directions) possibles par route
        routes = [ 
            (torch.tensor([0, self.center - 1]), torch.tensor([1, 0])), # Nord -> Sud
            (torch.tensor([self.grid_size-1, self.center]), torch.tensor([-1, 0])), # Sud -> Nord
            (torch.tensor([self.center, 0]), torch.tensor([0, 1])), # Ouest -> Est
            (torch.tensor([self.center - 1, self.grid_size-1]), torch.tensor([0, -1])) # Est -> Ouest 
            ]
        
        pos, direction = random.choice(routes)

        # on vérifie si la case est libre pour cette nouvelle voiture
        for i in range(self.max_agents):
            if self.active[i] and torch.equal(self.positions[i], pos):
                return
        
        for i in range (self.max_agents):
            if not self.active[i]:
                self.positions[i] = pos 
                self.directions[i] = direction 
                self.active[i] = True
                self.age[i] = 0 
                self.route_id[i] = random.randint(0, 2)     # 0: tout droit, 1: à gauche, 2: à droite
                self.has_turned[i] = False
                self.num_agents += 1
                return

    
    def apply_turn(self, i):
        """
        Applique le changement de direction (et donc de route et de voie) à un agent i
        quand il est au carrefour.
        """

        if self.route_id[i] == 0:
            self.has_turned[i] = True
            return
        
        if self.has_turned[i]:
            return
        
        x, y = self.positions[i].tolist()
        pos = (x, y)
        dx, dy = self.directions[i].tolist()

        # Cases du carrefour
        A = (self.center - 1, self.center - 1) 
        B = (self.center - 1, self.center) 
        C = (self.center, self.center - 1) 
        D = (self.center, self.center)

        # Voiture venant du Nord (direction Sud)
        if (dx, dy) == (1, 0):
            if self.route_id[i] == 2 and pos == A: # tourner à droite, -> Ouest
                self.directions[i] = torch.tensor([0, -1], device=self.device)
                self.has_turned[i] = True
            elif self.route_id[i] == 1 and pos == C: # tourner à gauche, -> Est
                self.directions[i] = torch.tensor([0, 1], device=self.device)
                self.has_turned[i] = True
        
        # Voiture venant du Sud (direction Nord)
        elif (dx, dy) == (-1, 0):
            if self.route_id[i] == 2 and pos == D: # tourner à droite, -> Est
                self.directions[i] = torch.tensor([0, 1], device=self.device)
                self.has_turned[i] = True
            elif self.route_id[i] == 1 and pos == B: # tourner à gauche, -> Ouest
                self.directions[i] = torch.tensor([0, -1], device=self.device)
                self.has_turned[i] = True
        
       # Voiture venant de l'Ouest (direction Est)
        elif (dx, dy) == (0, 1):
            if self.route_id[i] == 2 and pos == C: # tourner à droite, -> Sud
                self.directions[i] = torch.tensor([1, 0], device=self.device)
                self.has_turned[i] = True
            elif self.route_id[i] == 1 and pos == D: # tourner à gauche, -> Nord
                self.directions[i] = torch.tensor([-1, 0], device=self.device)
                self.has_turned[i] = True
            
        # Voiture venant de l'Est (direction Ouest)
        elif (dx, dy) == (0, -1):
            if self.route_id[i] == 2 and pos == B: # tourner à droite, -> Nord
                self.directions[i] = torch.tensor([-1, 0], device=self.device)
                self.has_turned[i] = True
            elif self.route_id[i] == 1 and pos == A: # tourner à gauche, -> Sud
                self.directions[i] = torch.tensor([1, 0], device=self.device)
                self.has_turned[i] = True


    def get_obs(self):
        """
        Observation locale : distance au centre + actif
        """
        center = torch.tensor([self.grid_size//2, self.grid_size//2], device=self.device) 
        dxdy = center - self.positions # (max_agents, 2) 
        obs = torch.cat([dxdy.float(), 
                         self.directions.float(), 
                         self.active.unsqueeze(-1).float()], 
                         dim=-1) 
        return obs


    def step(self, actions):
        """
        actions: (N,) BREAK=0, GAS=1
        """
        self.last_actions = actions.clone() # pour changer la forme des voitures à l'arrrêt pendant la simulation
        self.t += 1
        reward = 0.0

        ncollision_t = 0

        self.arriving_car()
        self.age[self.active] += 1

        # Appliquer le changement de direction au niveau du carrefour
        for i in range(self.max_agents): 
            if not self.active[i]: 
                continue

            if self.has_turned[i]:
                continue
            # si l'agent est bien au niveau du carrefour
            x, y = self.positions[i] 
            if (x == self.center or x == self.center - 1) and (y == self.center or y == self.center - 1): 
                self.apply_turn(i)

        # Déplacements des voitures
        for i in range(self.max_agents):
            if self.active[i] and actions[i] == GAS:
                self.positions[i] += self.directions[i]

        # Collision détectée?
        for i in range(self.max_agents):
            for j in range(i+1, self.max_agents):
                if self.active[i] and self.active[j]:
                    if torch.equal(self.positions[i], self.positions[j]):
                        # self.done = True
                        # reward -= 10.0
                        ncollision_t += 1
                        self.collision_count += 1
                        # return self.get_obs(), reward, self.done

        # Sortie de la grille ?
        for i in range(self.max_agents):
            if self.active[i]:
                x, y = self.positions[i]
                if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                    reward += 1.0
                    self.active[i] = False
                    self.age[i] = 0 
                    self.has_turned[i] = False
                    self.num_agents -= 1


        # Succès ou pas ?
        # if not self.active.any():
        #     self.done = True
            # return self.get_obs(), 1.0, self.done

        if self.t >= self.max_steps:
            self.done = True
        #     return self.get_obs(), 0.0, True

        # Calcul de la reward global au temps t
        reward = ncollision_t * (-10)
        reward += torch.sum(-0.01 * self.age[self.active].float()).item()
        self.global_reward += reward

        return self.get_obs(), reward, self.done


    def render(self):
        """
        """
        if not self.render_mode:
            return
        
        self.colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "pink", "brown"]

        self.canvas.delete("all") 
        cell = 520 // self.grid_size

        # Grille
        for i in range(self.grid_size):
            for j in range(self.grid_size):

                # Détection si la case est une route
                is_vertical_road = (j == self.center - 1 or j == self.center) 
                is_horizontal_road = (i == self.center - 1 or i == self.center)

                if is_vertical_road or is_horizontal_road: 
                    color = "#555555"
                else: 
                    color = "#209408"

                self.canvas.create_rectangle(
                    j*cell, i*cell, (j+1)*cell, (i+1)*cell,
                    outline = "white",
                    fill=color
                )

        # Affichage du compteur de collisions
        self.canvas.create_text(10, 10, anchor="nw", 
                                text=f"Collisions : {self.collision_count}", 
                                fill="black", font=("Arial", 16, "bold"))
        
        # Affichage de la reward globale
        self.canvas.create_text( 10, 40, anchor="nw", 
                                text=f"Reward global : {self.global_reward:.2f}", 
                                fill="black", font=("Arial", 16, "bold") )

        # Voitures
        for i in range(self.max_agents): 
            if not self.active[i]: 
                continue
                
            x = int(self.positions[i, 0].item()) 
            y = int(self.positions[i, 1].item())

            if hasattr(self, "last_actions") and self.last_actions[i] == BREAK: # si s'arrête, alors en forme de rectangle
                self.canvas.create_rectangle( 
                    y*cell+5, x*cell+5, y*cell+cell-5, x*cell+cell-5, 
                    fill=self.colors[i % len(self.colors)])
            else: # si avance, alors en forme de rond
                self.canvas.create_oval( 
                    y*cell+5, x*cell+5, y*cell+cell-5, x*cell+cell-5, 
                    fill=self.colors[i % len(self.colors)])

        self.window.update()
        time.sleep(0.3)


def main():
    env = TrafficJunction(grid_size=14, 
                             max_agents=10, 
                             p_arrive=0.3, 
                             device="cpu", 
                             render_mode=True)

    obs = env.reset()
    done = False
    reward = 0.0

    while not done:
        actions = torch.randint(0, 2, (env.max_agents,))
        obs, reward, done = env.step(actions)
        env.render()
        # time.sleep(0.3)

    print("Reward:", reward, " Success ? " , env.collision_count == 0)

if __name__ == "__main__":
    main()