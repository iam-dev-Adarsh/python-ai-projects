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

