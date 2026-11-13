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

BORDER_COLOR = (255,255,255,255) # Color to crash on hit

current_generation = 0 # Generationcounter

class Car:
    def  __init__(self):
        # load car sprite and rotate
        self.sprite = pygame.image.load('car.png').convert()
    # convert aids in speeding up a lot!
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X,CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [830,920]  # initial starting position
        self.angle = 0
        self.speed = 0

        self.spped_test = False # F;ag for default speed later on

        self.center = [self.position[0] + CAR_SIZE_X/2,self.position[1] + CAR_SIZE_Y / 2] # calculate centre 

        self.radars = [] # List for sensors and radars
        self.drawing_radars = [] # radars to be drawn

        self.alive = True # to check if whether the car has crashed or not

        self.distance = 0 # Dstance driven
        self.time = 0  # Time passed   

    def draw(self,screen):
        screen.blit(self.rotated_sprite, self.position) # draw sprite (track)
        self.draw_radar(screen) # can be useful for sensors but its optional

    def draw_radar(self,screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0,255,0), self.center,position,1 )
            pygame.draw.circle(screen, (0,255,0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # if any corner touches border color - > crash
            # Assumes rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self,degree,game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While we dont hit BORDER COLOR and length < 300 -> should go further
        while not game_map.get_at((x,y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

            # calculate distance to border and append radars list
            dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1] , 2)))
            self.radars.append([(x,y) , dist])

                        


            