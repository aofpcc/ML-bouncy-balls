import pygame
import neat
import os
import pickle

max_score = 2000
gen = 0
DRAW_LINES = True
the_pattern = None
FPS = 150
every = 5


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


def eval_genomes(genomes, eval_config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    global gen
    gen += 1

    game_play = GamePlay(FPS, gen, DRAW_LINES, the_pattern)
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
        net = neat.nn.FeedForwardNetwork.create(genome, eval_config)
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
        if game_play.score > max_score:
            break

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # game_play.check_event(event

        for x, player in enumerate(players):  # give each bird a fitness of 0.1 for each frame it stays alive
            if player.state == PlayerState.DEAD:
                continue

            # ge[x].fitness += 0.1
            ge[x].fitness = game_play.score
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


def train_model(config_file, pattern):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    global the_pattern
    the_pattern = pattern

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-439')

    prefix = './pattern-' + str(pattern) + '/';

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(every, filename_prefix=prefix + 'neat-checkpoint-'))

    # Run for up to 200 generations.
    winner = p.run(eval_genomes, 20000)

    with open(prefix + 'winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


def run_model(config_file, pattern):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    game_play = GamePlay(FPS, draw_line=DRAW_LINES, the_pattern=pattern)
    fpsClock = game_play.get_fps_clock

    with open('./pattern-' + str(pattern) + '/winner.pkl', 'rb') as input_file:
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


def play_by_yourself(config_file, pattern):
    FPS = 20
    DRAW_LINES = False
    game_play = GamePlay(FPS, draw_line=DRAW_LINES, the_pattern=pattern)
    fpsClock = game_play.get_fps_clock

    nets = []
    players = []
    game_play.prepare()

    game_play.add_main_player()

    while game_play.state != GameState.ALL_DEAD:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            game_play.check_event(event)

        game_play.draw()
        pygame.display.flip()
        pygame.display.update()
        fpsClock.tick(FPS)
    print('Best Score: ', game_play.score)


def play_against_model(config_file, pattern):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    FPS = 20
    DRAW_LINES = False
    game_play = GamePlay(FPS, draw_line=DRAW_LINES, the_pattern=pattern)
    fpsClock = game_play.get_fps_clock

    with open('./pattern-' + str(pattern) + '/winner.pkl', 'rb') as input_file:
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

    game_play.add_main_player()

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

        if len(game_play.dead_players) == 1:
            break

    print('Best Score: ', game_play.score)

    if game_play.main_player.state == PlayerState.DEAD:
        print("AI wins")
    else:
        print("You wins")


switcher = {
    1: train_model,
    2: run_model,
    3: play_by_yourself,
    4: play_against_model
}

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')

    main_menu = 'Choose menu: \n1.Train Model\n2.Run Model\n3.Play by yourself\n4.Play against Model\nChosen input: '
    # run(config_path)

    chosen_menu = None

    while True:
        chosen_menu = input(main_menu)
        if chosen_menu in ['1', '2', '3', '4']:
            break
        print("Incorrect input! Please try again!!\n")

    pattern_menu = 'What pattern?: \n1.A\n2.B\n3.C\nChosen input: '
    while True:
        chosen_pattern = input(pattern_menu)
        if chosen_pattern in ['1', '2', '3']:
            break
        print("Incorrect input! Please try again!!\n")

    chosen_pattern = {
        '1': 'a',
        '2': 'b',
        '3': 'c'
    }.get(chosen_pattern)

    from GameComponent import *

    pygame.init()
    pygame.display.set_caption("Bouncy Ball")

    chosen_menu = int(chosen_menu)
    func = switcher.get(chosen_menu)
    func(config_path, chosen_pattern)
