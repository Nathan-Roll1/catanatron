from catanatron.cli.cli_players import parse_cli_string
from catanatron.cli.play import GameConfigOptions, OutputOptions, play_batch

players = parse_cli_string("AB:2,AB:2")
game_config = GameConfigOptions(7, 10, "BASE")
play_batch(5, players, OutputOptions(), game_config, quiet=True)
