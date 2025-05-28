import pytest
from unittest import mock
import random


class BaseballPlayerStats:
    def __init__(self, name):
        self.name = name
        self.hits = 0
        self.at_bats = 0
        self.home_runs = 0
        self.runs_batted_in = 0
        self.walks = 0
        self.strikeouts = 0

    def record_hit(self, home_run=False, rbi=0):
        self.hits += 1
        self.at_bats += 1
        self.runs_batted_in += rbi
        if home_run:
            self.home_runs += 1

    def record_at_bat(self):
        self.at_bats += 1

    def record_walk(self):
        self.walks += 1

    def record_strikeout(self):
        self.strikeouts += 1
        self.at_bats += 1

    def batting_average(self):
        if self.at_bats == 0:
            return 0.0
        return self.hits / self.at_bats

    def __str__(self):
        return (f"{self.name}: AVG={self.batting_average():.3f}, "
                f"HR={self.home_runs}, RBI={self.runs_batted_in}, "
                f"BB={self.walks}, SO={self.strikeouts}")


class BaseballGameSimulator:
    def __init__(self, players):
        self.players = {player.name: player for player in players}
        self.inning = 1
        self.score = {player.name: 0 for player in players}

    def simulate_play(self, player_name):
        player = self.players[player_name]
        play = random.choices(
            ['hit', 'home_run', 'walk', 'strikeout', 'out'],
            weights=[0.2, 0.05, 0.1, 0.2, 0.45],
            k=1
        )[0]
        player.record_at_bat()

        if play == 'hit':
            player.record_hit(rbi=random.randint(0, 2))
        elif play == 'home_run':
            player.record_hit(home_run=True, rbi=random.randint(0, 4))
            self.score[player_name] += 1
        elif play == 'walk':
            player.record_walk()
        elif play == 'strikeout':
            player.record_strikeout()
        else:
            raise ValueError("Invalid play type")

        return play

    def simulate_inning(self):
        for player_name in self.players:
            self.simulate_play(player_name)
        self.inning += 1


###########################################################
# Tests for BaseballPlayerStats and BaseballGameSimulator #
###########################################################

def test_baseball_player_stats_basic():
    player = BaseballPlayerStats("Test Player")
    assert player.batting_average() == 0.0
    player.record_hit()
    assert player.hits == 1
    assert player.at_bats == 1
    assert player.batting_average() == 1.0
    player.record_at_bat()
    assert player.at_bats == 2
    assert player.batting_average() == 0.5
    player.record_hit(home_run=True, rbi=2)
    assert player.home_runs == 1
    assert player.runs_batted_in == 2
    assert player.hits == 2
    assert player.at_bats == 3
    player.record_walk()
    assert player.walks == 1
    player.record_strikeout()
    assert player.strikeouts == 1
    assert player.at_bats == 4

def test_baseball_game_simulator_simulate_play_hit():
    player = BaseballPlayerStats("Player1")
    sim = BaseballGameSimulator([player])

    with mock.patch("random.choices", return_value=["hit"]), \
         mock.patch("random.randint", return_value=1):
        play = sim.simulate_play("Player1")
        assert play == "hit"
        assert player.hits == 1
        assert player.at_bats == 2  # record_at_bat + record_hit
        assert player.runs_batted_in == 1

def test_baseball_game_simulator_simulate_play_home_run(mocker):
    player = BaseballPlayerStats("Player2")
    sim = BaseballGameSimulator([player])

    with mocker.patch("random.choices", return_value=["home_run"]), \
         mocker.patch("random.randint", return_value=3):
        play = sim.simulate_play("Player2")
        assert play == "home_run"
        assert player.home_runs == 1
        assert player.hits == 1
        assert player.at_bats == 2  # record_at_bat + record_hit
        assert player.runs_batted_in == 3
        assert sim.score["Player2"] == 1

def test_baseball_game_simulator_simulate_play_walk(mocker):
    sim = BaseballGameSimulator([player])
    player = mocker.Mock()

    with mocker.patch("random.choices", return_value=["walk"]):
        play = sim.simulate_play("Player3")
        assert play == "walk"
        assert player.walks == 1
        assert player.at_bats == 1  # record_at_bat only

def test_baseball_game_simulator_simulate_play_strikeout(mocker):
    player = BaseballPlayerStats("Player4")
    sim = BaseballGameSimulator([player])

    with mocker.patch("random.choices", return_value=["strikeout"]):
        play = sim.simulate_play("Player4")
        assert play == "strikeout"
        assert player.strikeouts == 1
        assert player.at_bats == 2  # record_at_bat + record_strikeout
