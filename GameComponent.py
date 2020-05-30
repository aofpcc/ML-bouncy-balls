#!/usr/bin/env python
from enum import Enum

import pygame
import sys
import time
import random
import numpy as np
from random import *
from button import Button
from pygame.locals import *
import neat
import math

pygame.init()

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
GREY = (50, 50, 50)
WHITE = (255, 255, 255)
ORANGE = (255, 180, 0)

BUTTON_STYLE = {"hover_color": GREY,
                "clicked_color": WHITE,
                "clicked_font_color": BLACK,
                "font": pygame.font.SysFont("monospace", 30)}

fpsClock = pygame.time.Clock()

SCREEN_WIDTH, SCREEN_HEIGHT = 300, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
surface = pygame.Surface(screen.get_size())
surface = surface.convert()
surface.fill((255, 255, 255))
clock = pygame.time.Clock()

pygame.key.set_repeat(1, 40)

GRID_SIZE = 1
GRID_WIDTH = SCREEN_WIDTH / GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT / GRID_SIZE

GRAVITY = 2.5
SCALE_GRAVITY = GRID_HEIGHT * GRAVITY * 0.001

# print("SCALE_GRAVITY = {0}".format(SCALE_GRAVITY))

screen.blit(surface, (0, 0))

OBSTACLE_SPEED = GRID_HEIGHT * 0.01
OBSTACLE_MIN_WIDTH = GRID_WIDTH / 3.4
OBSTACLE_MAX_WIDTH = GRID_WIDTH / 2

text_font = pygame.font.SysFont('monospace', 30)


class GameState(Enum):
    MENU = 1
    PLAYING = 2
    WAITING = 3
    ALL_DEAD = 4


class PlayerState(Enum):
    ALIVE = 1
    DEAD = 2


def draw_box(surf, color, pos, dim):
    r = pygame.Rect((int(pos[0] - dim[0] / 2 + 1), pos[1]), dim)
    pygame.draw.rect(surf, color, r)


def draw_circle(surf, color, pos, radius):
    pygame.draw.circle(surf, color, pos, radius)


def draw_button(surf, color, pos, dim, text):
    r = pygame.Rect(pos, dim)
    pygame.draw.rect(surf, color, r)


class Component:
    ID = 0

    def __init__(self, the_screen, position):
        Component.ID += 1
        self.id = Component.ID
        self.screen = the_screen
        self.position = position
        self.screen_size = self.screen.get_size()

    def draw(self, surf):
        pass

    def is_out(self):
        pass

    def set_position(self, position):
        self.position = (int(position[0]), int(position[1]))


class Obstacle(Component):
    def __init__(self, the_screen, position, radius, game_play):
        super().__init__(the_screen, position)
        self.radius = radius
        self.color = (0, 0, 0)
        self.game_play = game_play
        self.velocity = (0, OBSTACLE_SPEED)
        self.randomize()

    def randomize(self):
        radius = self.radius
        # int(uniform(radius, GRID_WIDTH - radius))
        x = int(GRID_WIDTH//4)
        # i = randrange(5)
        i = self.game_play.get_current_obstacle_pattern() - 1
        self.set_position((i * x, self.position[1]))

        # print(self.position, self.radius)
        # print("----------------------------")

    def update(self):
        self.set_position(np.add(self.position, self.velocity))

        if self.is_out():
            self.randomize()
            self.set_position((self.position[0], -self.radius))

    #         - self.game_play.obstacle_gap *
    #                                (self.game_play.number_of_obstacles - 2))

    def draw(self, surf):
        draw_circle(surf, self.color, self.position, self.radius)

    def is_out(self):
        return self.position[1] - self.radius > self.screen_size[1]

    def touch(self, players):
        for player in players:
            if player.state == PlayerState.DEAD:
                continue
            a = self.distance(player) < player.radius + self.radius - 0.1
            if a:
                player.dead()

    def distance(self, player):
        return math.sqrt((self.position[0] - player.position[0]) ** 2 + (self.position[1] - player.position[1]) ** 2)


class Player(Component):
    def __init__(self, the_screen, position, player_radius, game_play, border_width=0, jump_power=8 * SCALE_GRAVITY):
        super().__init__(the_screen, position)
        self.radius = player_radius
        self.game_play = game_play
        self.state = PlayerState.ALIVE
        # velocity => (x,y)
        self.color = pygame.Color(50 + int(150 * random()), 50 + int(150 * random()), 50 + int(150 * random()))
        self.velocity = (0, 0)
        self.border_width = border_width
        self.jump_power = jump_power
        self.radius_without_border = self.radius - border_width
        self.dead_score = 0
        self.last_jump = -7
        self.dead_time = 0

    def jump(self, is_right=False):
        if self.game_play.current_fps - self.last_jump < 7:
            return
        self.last_jump = self.game_play.current_fps
        if self.state == PlayerState.DEAD:
            return
        # print("Player id {0} jump".format(self.id))
        x_speed = -0.025 * GRID_WIDTH
        if is_right:
            x_speed *= -1
        self.velocity = (x_speed, -self.jump_power)
        # self.velocity = (0, -self.jump_power)

    def is_out(self):
        return self.position[1] + self.radius >= self.screen_size[1] or \
               self.position[1] - self.radius < 0

    def touch_boundary(self):
        return self.position[0] + self.radius > self.screen_size[0] \
               or self.position[0] - self.radius < 0

    def update_position(self):
        self.set_position(np.add(self.position, self.velocity))

    def update(self):
        if self.game_play.state in [GameState.ALL_DEAD, GameState.WAITING]:
            return
        if self.state == PlayerState.DEAD:
            self.velocity = (0, OBSTACLE_SPEED)
            self.update_position()
            return
        self.velocity = np.add(self.velocity, (0, SCALE_GRAVITY))
        self.update_position()

        if self.is_out():
            # self.set_position((self.position[0], GRID_HEIGHT - self.radius))
            self.dead()
            # print("DEAD")
            return

        if self.touch_boundary():
            if self.position[0] - self.radius < 0:
                self.set_position((0 + self.radius, self.position[1]))
            else:
                self.set_position((GRID_WIDTH - self.radius, self.position[1]))
            self.velocity[0] *= -1

    def dead(self):
        if self.state == PlayerState.DEAD:
            return
        self.state = PlayerState.DEAD
        # Add player to the list of dead players
        self.game_play.dead(self)
        self.dead_score = self.game_play.get_score()
        self.dead_time = time.time()

    def start(self):
        self.state = PlayerState.ALIVE

    def draw(self, surf):
        self.update()
        draw_circle(surf, BLACK, self.position, self.radius)
        draw_circle(surf, self.color, self.position, self.radius_without_border)

        # pygame.draw.line(surf, BLUE, (int(x + radius), 0), (int(x + radius), GRID_HEIGHT), 1)
        # pygame.draw.line(surf, BLUE, (int(x - radius), 0), (int(x - radius), GRID_HEIGHT), 1)
        #
        # pygame.draw.line(surf, BLUE, (0, int(y + radius)), (GRID_WIDTH, int(y + radius)), 1)
        # pygame.draw.line(surf, BLUE, (0, int(y - radius)), (GRID_WIDTH, int(y - radius)), 1)


class AI(Player):
    def __init__(self, screen, position, player_radius, game_play):
        super().__init__(screen, position, player_radius, game_play)

    # def update(self):
    #     super().update()
    #     y = self.position[1] + self.radius
    #     diff = (self.screen_size[1] - (self.screen_size[1] - y)) / self.screen_size[1]
    #     if random() <= 0.15 * diff:
    #         self.jump(random() >= 0.5)


class GamePlay(object):
    def __init__(self, fps, gen=None, draw_line=False, the_pattern=None):
        self.current_fps = 0
        self.FPS = fps
        self.state = GameState.MENU
        self.gen = gen

        self.score = 0
        self.start_time = 0

        self.players = []
        self.dead_players = []
        self.main_player = None

        self.screen = screen
        self.fpsClock = fpsClock
        self.surface = surface
        self.screen_rect = self.screen.get_rect()

        self.count = 0
        self.play_button = Button((0, 0, 200, 50), BLACK, self.play_button_click,
                                  text="Play", **BUTTON_STYLE)
        self.play_button.rect.center = (self.screen_rect.centerx, 100)

        self.y_scale = int(GRID_HEIGHT // 4)
        self.x_scale = int(GRID_WIDTH // 4)
        self.starting_y = 3 * self.y_scale
        self.player_radius = int(GRID_WIDTH // 12)

        self.obstacles = []
        self.number_of_obstacles = 3
        self.obstacle_gap = GRID_HEIGHT // self.number_of_obstacles
        self.obstacle_radius = int(GRID_WIDTH // 12)

        self.player_init_position = (2 * self.x_scale, self.starting_y)
        self.draw_line = draw_line

        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT

        self.obstacle_patterns = [
            # [1, 2, 3, 4, 5], # pattern a
            # [5, 4, 3, 2, 1], # pattern b
            # [2, 4, 2, 4, 3], # pattern c
            # [1, 5, 2, 4, 3], # pattern d
        ]

        if the_pattern is None or the_pattern == 'a':
            self.obstacle_patterns.append([1, 2, 3, 4, 5])
        elif the_pattern == 'b':
            self.obstacle_patterns.append([5, 4, 3, 2, 1])
        elif the_pattern == 'c':
            self.obstacle_patterns.append([2, 4, 2, 4, 3])

        self.total_patterns = len(self.obstacle_patterns)

        self.current_patterns = []

    def get_score(self):
        return self.score

    def set_up(self):
        self.score = 0
        y_scale = int(GRID_HEIGHT // 4)
        x_scale = int(GRID_WIDTH // 4)
        #
        self.players = []
        self.dead_players = []
        # ai = AI(self.screen, self.player_init_position, self.player_radius, self)
        # self.players.append(ai)
        # ai = AI(self.screen, self.player_init_position, self.player_radius, self)
        # self.players.append(ai)
        # self.add_main_player()

        # obstacles
        self.obstacles = []
        obstacle_height = int(GRID_HEIGHT // 100)
        for i in range(self.number_of_obstacles):
            obstacle = Obstacle(self.screen, (2 * x_scale, -self.obstacle_radius -i * self.obstacle_gap),
                                self.obstacle_radius, self)
            self.obstacles.append(obstacle)

    def get_current_obstacle_pattern(self):
        if len(self.current_patterns) == 0:
            pos = randrange(self.total_patterns)
            self.current_patterns += self.obstacle_patterns[pos]
            # here
        result = self.current_patterns[0]
        self.current_patterns = self.current_patterns[1:]
        return result

    def add_player(self, player):
        self.players.append(player)

    def add_main_player(self):
        self.main_player = Player(self.screen, self.player_init_position, self.player_radius, self, 2)
        self.add_player(self.main_player)

    def dead(self, player):
        self.dead_players.append(player)

    def prepare(self):
        self.set_up()
        self.state = GameState.WAITING

    def play_button_click(self):
        self.prepare()
        self.add_main_player()

    def start(self):
        self.start_time = time.time()
        self.state = GameState.PLAYING
        for player in self.players:
            player.start()

    def draw_menu(self, surf):
        self.play_button.update(surf)

    def check_event(self, event):
        self.play_button.check_event(event)
        if event.type == KEYDOWN:
            self.key_down(event.key)

    def key_down(self, key):
        self.count += 1
        if key in [K_o]:
            if self.state not in [GameState.MENU, GameState.ALL_DEAD]:
                return
            self.prepare()
        if key in [K_p]:
            if self.state != GameState.WAITING:
                return
            self.start()
        if key in [K_RIGHT, K_LEFT]:
            if self.state == GameState.WAITING:
                self.start()
            if self.state != GameState.PLAYING:
                return
            self.main_player.jump(key == K_RIGHT)

    def draw(self):
        self.current_fps += 1

        self.surface.fill((255, 255, 255))

        for x in range(0, int(GRID_WIDTH), 25):
            pygame.draw.line(self.surface, (200, 200, 200), (x, 0), (x, int(GRID_HEIGHT)), 1)
        for y in range(0, int(GRID_HEIGHT), 25):
            pygame.draw.line(self.surface, (200, 200, 200), (0, y), (int(GRID_WIDTH), y), 1)

        t = time.time()

        if self.state in [GameState.PLAYING, GameState.WAITING, GameState.ALL_DEAD]:
            if len(self.players) == len(self.dead_players):
                self.state = GameState.ALL_DEAD
            if self.state == GameState.PLAYING:
                self.score = self.current_fps / 7
            for player in self.players:
                if player.state == PlayerState.DEAD and t - player.dead_time > 1:
                    continue
                if self.draw_line:
                    obstacles = self.closest_obstacles(player)
                    for obstacle in obstacles:
                        if obstacle[1] > 1000:
                            continue
                        pygame.draw.line(self.surface, RED, player.position, obstacle[0], 1)
                player.draw(self.surface)
            for obstacle in self.obstacles:
                if self.state == GameState.PLAYING:
                    obstacle.update()
                    obstacle.touch(self.players)
                obstacle.draw(self.surface)

        if self.state in [GameState.MENU
                          # ,GameState.ALL_DEAD
                          ]:
            self.draw_menu(self.surface)

        self.screen.blit(self.surface, (0, 0))

        for player in self.dead_players:
            if player.state == PlayerState.DEAD and t - player.dead_time > 1:
                continue
            text_surface = text_font.render("{:.1f}".format(player.dead_score), True, (255, 255, 255), (0, 25, 0))
            self.screen.blit(text_surface, (player.position[0] - player.radius, player.position[1] - 2.5 * player.radius))

        text_surface = text_font.render("{:.1f}" .format(self.score), True, (255, 255, 255), (0, 0, 0))
        score_y = 0
        if self.gen is not None:
            score_y = 35
        self.screen.blit(text_surface, (0, score_y))

        if self.gen is not None:
            text_surface = text_font.render("Gen: {0}".format(self.gen), True, (255, 255, 255), (0, 0, 0))
            self.screen.blit(text_surface, (0, 0))

    def closest_obstacles(self, player):
        obstacle_list = []
        max_dist = 99999
        max_position = (GRID_WIDTH, -99999)
        radius = self.obstacle_radius
        for obstacle in self.obstacles:
            position = obstacle.position
            if position[1] - radius < GRID_HEIGHT or position[1] + radius > 0:
                d = obstacle.distance(player)
                obstacle_list.append((obstacle.position, d))
            else:
                obstacle_list.append((max_position, 99999))
        o_len = 3
        for i in range(0, len(obstacle_list) - o_len):
            obstacle_list.append((max_position, 99999))
        obstacle_list.sort(key=lambda tup: tup[1])
        return obstacle_list[0:o_len]

    @property
    def get_fps_clock(self):
        return self.fpsClock
