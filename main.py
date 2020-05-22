from GameComponent import *
import pygame
import neat
import os
import pickle

pygame.init()
DRAW_LINES = True
pygame.display.set_caption("Bouncy Ball")

gen = 0


def jump_or_not(player, g_play, network):
    p_position = player.position
    obstacles = g_play.closest_obstacles(player)
    o_1 = obstacles[0]
    o_2 = obstacles[1]
    o_3 = obstacles[2]
    # diff = time.time() - player.last_jump

    output = network.activate(
        (p_position[0], p_position[1],
         g_play.grid_width - p_position[0], g_play.grid_height - p_position[1],
         o_1[0][0] - p_position[0], o_1[0][1] - p_position[1],
         o_2[0][0] - p_position[0], o_2[0][1] - p_position[1],
         o_3[0][0] - p_position[0], o_3[0][1] - p_position[1],
         )
    )

    if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5
        # jump
        player.jump(output[1] > 0)


def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    global gen
    gen += 1

    FPS = 15
    game_play = GamePlay(FPS, gen, DRAW_LINES)
    fpsClock = game_play.get_fps_clock

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    players = []
    ge = []
    # print(len(genomes))
    game_play.prepare()
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        new_player = AI(game_play.screen, game_play.player_init_position, game_play.player_radius, game_play)
        players.append(new_player)
        game_play.add_player(new_player)
        ge.append(genome)

    game_play.start()

    grid_width = game_play.grid_width
    grid_height = game_play.grid_height

    # print(len(players))

    while game_play.state != GameState.ALL_DEAD:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # game_play.check_event(event

        for x, player in enumerate(players):  # give each bird a fitness of 0.1 for each frame it stays alive
            if player.state == PlayerState.DEAD:
                continue

            # ge[x].fitness += 0.1
            network = nets[players.index(player)]
            jump_or_not(player, game_play, network)

        for player in players:
            if player.state == PlayerState.DEAD:
                ge[players.index(player)].fitness = player.dead_score
                nets.pop(players.index(player))
                ge.pop(players.index(player))
                players.pop(players.index(player))

        game_play.draw()

        pygame.display.flip()
        pygame.display.update()
        fpsClock.tick(FPS)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    # p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-262')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(20))

    # Run for up to 200 generations.
    winner = p.run(eval_genomes, 2)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    # run(config_path)

    config_file = config_path
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    FPS = 60
    game_play = GamePlay(FPS, draw_line=DRAW_LINES)
    fpsClock = game_play.get_fps_clock

    with open('winner.pkl', 'rb') as input_file:
        genome = pickle.load(input_file)

    nets = []
    players = []

    game_play.prepare()

    for i in range(1):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        new_ai = AI(game_play.screen, game_play.player_init_position, game_play.player_radius, game_play)
        players.append(new_ai)
        game_play.add_player(new_ai)

    game_play.start()

    while game_play.state != GameState.ALL_DEAD:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            game_play.check_event(event)

        for player in players:
            network = nets[players.index(player)]
            jump_or_not(player, game_play, network)

        for player in players:
            if player.state == PlayerState.DEAD:
                nets.pop(players.index(player))
                players.pop(players.index(player))

        game_play.draw()
        pygame.display.flip()
        pygame.display.update()
        fpsClock.tick(FPS)

    print('Best Score: ', game_play.score)
