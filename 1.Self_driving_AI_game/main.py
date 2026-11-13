import math
import random
import sys
import os
import neat
import pygame

# Constants
WIDTH = 1600
HEIGHT = 880

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

# Border color (white background/wall = crash)
BORDER_COLOR = (255, 255, 255, 255)

current_generation = 0


class Car:
    def __init__(self):
        # Load car sprite
        self.sprite = pygame.image.load('car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [830, 920]  # Starting position
        self.angle = 0
        self.speed = 0
        self.speed_set = False  # Fixed variable name

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        self.radars = []
        self.alive = True
        self.distance = 0
        self.time = 0

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 4)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            px, py = int(point[0]), int(point[1])
            # Check bounds to prevent IndexError crash
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                if game_map.get_at((px, py)) == BORDER_COLOR:
                    self.alive = False
                    break
            else:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while length < 300:
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if game_map.get_at((x, y)) == BORDER_COLOR:
                    break
            else:
                break
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        
        # Position updates with bounds clamping (min/max)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], HEIGHT - 120)

        self.distance += self.speed
        self.time += 1

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate 4 corners of car
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(game_map)
        self.radars.clear()

        # 5 Distance Radars (-90, -45, 0, 45, 90)
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            if i < 5:
                return_values[i] = int(radar[1] / 30)
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        return rotated_image.subsurface(rotated_rectangle).copy()


def run_simulation(genomes, config):
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving Car Simulation (NEAT AI)")

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()  # Fixed Clock capitalization
    generation_font = pygame.font.SysFont('Arial', 32)
    alive_font = pygame.font.SysFont('Arial', 24)
    game_map = pygame.image.load('map.png').convert()

    global current_generation
    current_generation += 1

    counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Fixed pygame.QUIT
                sys.exit(0)

        # Feed radar sensor inputs to the neural net
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow down
            else:
                car.speed += 2  # Speed up

        # Update alive cars & compute fitness
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        # Stop generation if taking too long (stuck cars)
        if counter == 30 * 40:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Draw HUD stats
        gen_text = generation_font.render(f'Generation: {current_generation}', True, (0, 0, 0))
        screen.blit(gen_text, (50, 40))

        alive_text = alive_font.render(f'Still Alive: {still_alive}', True, (0, 0, 0))
        screen.blit(alive_text, (50, 80))

        pygame.display.flip()
        clock.tick(60)  # Fixed tick method


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