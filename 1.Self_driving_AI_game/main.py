import math
import random
import sys
import os
import neat
import pygame

# Screen Constants
WIDTH = 1600
HEIGHT = 880

# Car size (fits inside the road)
CAR_SIZE_X = 35
CAR_SIZE_Y = 18

# Border color (pure white outside the track = crash)
BORDER_COLOR = (255, 255, 255, 255)

current_generation = 0

# Initialize Pygame and Screen ONCE globally to prevent window flickering
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation (NEAT AI)")
clock = pygame.time.Clock()
generation_font = pygame.font.SysFont('Arial', 32, bold=True)
alive_font = pygame.font.SysFont('Arial', 24)

# Load and scale map once globally to match (WIDTH, HEIGHT)
raw_map = pygame.image.load('map.png').convert()
game_map = pygame.transform.scale(raw_map, (WIDTH, HEIGHT))


class Car:
    def __init__(self):
        # Load and scale car sprite
        self.sprite = pygame.image.load('car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))

        # Safe center position on the asphalt road on the top straight
        self.center = [600.0, 85.0]
        self.angle = 180.0  # Facing left along the long straight runway
        self.speed = 8.0

        # Rotate sprite and get centered rectangle (no subsurface bug!)
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.rect = self.rotated_sprite.get_rect(center=(int(self.center[0]), int(self.center[1])))

        self.radars = []
        self.alive = True
        self.distance = 0.0
        self.time = 0

    def draw(self, screen_surf):
        screen_surf.blit(self.rotated_sprite, self.rect.topleft)
        self.draw_radar(screen_surf)

    def draw_radar(self, screen_surf):
        for radar in self.radars:
            pos = radar[0]
            pygame.draw.line(screen_surf, (0, 255, 0), (int(self.center[0]), int(self.center[1])), pos, 1)
            pygame.draw.circle(screen_surf, (0, 255, 0), pos, 3)

    def check_collision(self, map_surf):
        self.alive = True
        for point in self.corners:
            px, py = int(point[0]), int(point[1])
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                pixel_color = map_surf.get_at((px, py))
                # Crash if touching pure white border (outside track)
                if pixel_color[:3] == (255, 255, 255):
                    self.alive = False
                    break
            else:
                self.alive = False
                break

    def check_radar(self, degree, map_surf):
        length = 0
        rad = math.radians(360.0 - (self.angle + degree))
        cos_val = math.cos(rad)
        sin_val = math.sin(rad)

        x = int(self.center[0] + cos_val * length)
        y = int(self.center[1] + sin_val * length)

        while length < 250:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pixel_color = map_surf.get_at((x, y))
                if pixel_color[:3] == (255, 255, 255):
                    break
            else:
                break
            length += 2
            x = int(self.center[0] + cos_val * length)
            y = int(self.center[1] + sin_val * length)

        dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        self.radars.append([(x, y), dist])

    def update(self, map_surf):
        # Update linear displacement
        rad = math.radians(360.0 - self.angle)
        self.center[0] += math.cos(rad) * self.speed
        self.center[1] += math.sin(rad) * self.speed

        self.distance += self.speed
        self.time += 1

        # Affine centered rotation without subsurface bounding box overflow
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.rect = self.rotated_sprite.get_rect(center=(int(self.center[0]), int(self.center[1])))

        # Calculate exact 4 corners of car
        half_w = CAR_SIZE_X / 2.0
        half_h = CAR_SIZE_Y / 2.0
        diag = math.sqrt(half_w ** 2 + half_h ** 2)

        alpha = math.degrees(math.atan2(half_h, half_w))
        angles = [alpha, 180.0 - alpha, 180.0 + alpha, 360.0 - alpha]

        self.corners = []
        for a in angles:
            corner_rad = math.radians(360.0 - (self.angle + a))
            cx = self.center[0] + math.cos(corner_rad) * diag
            cy = self.center[1] + math.sin(corner_rad) * diag
            self.corners.append([cx, cy])

        self.check_collision(map_surf)
        self.radars.clear()

        # 5 Distance Radar Rays (-90, -45, 0, 45, 90)
        for d in range(-90, 120, 45):
            self.check_radar(d, map_surf)

    def get_data(self):
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            if i < 5:
                return_values[i] = int(radar[1] / 25.0)
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2.0)


def run_simulation(genomes, config):
    global current_generation
    current_generation += 1

    nets = []
    cars = []

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0.0
        cars.append(Car())

    counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # Predict actions for each alive car
        for i, car in enumerate(cars):
            if car.is_alive():
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))
                if choice == 0:
                    car.angle += 8.0   # Steer Left
                elif choice == 1:
                    car.angle -= 8.0   # Steer Right
                elif choice == 2:
                    if car.speed - 1.0 >= 4.0:
                        car.speed -= 1.0  # Brake
                else:
                    if car.speed + 1.0 <= 12.0:
                        car.speed += 1.0  # Accelerate

        # Update physics & record fitness
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        # If all cars crashed, advance generation
        if still_alive == 0:
            break

        counter += 1
        # Generation timeout (e.g., 25 seconds at 60 FPS) to prevent stuck cars
        if counter >= 60 * 25:
            break

        # Render track & cars
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Draw HUD stats
        gen_text = generation_font.render(f'Generation: {current_generation}', True, (255, 255, 255))
        screen.blit(gen_text, (50, 40))

        alive_text = alive_font.render(f'Still Alive: {still_alive}', True, (0, 255, 0))
        screen.blit(alive_text, (50, 80))

        pygame.display.flip()
        clock.tick(0)


if __name__ == '__main__':
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(run_simulation, 1000)
