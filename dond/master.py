
from dataclasses import dataclass
import logging
import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemcore.clemgame.master import ParseError, RuleViolationError, GameError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from clemcore.utils import string_utils

from nltk.stem.snowball import SnowballStemmer

logger = logging.getLogger(__name__)


def format_item_set(counts: list[int], types: list[str], types_plural: list[str]) -> str:
    return ', '.join([
        str(count) + ' ' + (single if count == 1 else plural)
        for count, single, plural in zip(counts, types, types_plural)
    ])


def format_value_function(values: list[int], types: list[str]) -> str:
    return ', '.join([
        type + ': ' + str(value) for value, type in zip(values, types)
    ])


class DealOrNoSealPlayer(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = [
            'I will take nothing.', 'I will take everything.',
            '[Proposal:]', '[Proposal: 2 hats, 1 book]',
        ]

    def _custom_response(self, messages):
        word = self._custom_responses.pop(0)
        return f'{word}'


@dataclass
class GameState:
    max_rounds: int
    player_a_initial_prompt: str
    player_b_initial_prompt: str
    proposal_prompt_early: str
    proposal_prompt_timeout: str
    item_types: list[str]
    item_types_plural: list[str]
    item_counts: list[int]
    player_a_values: list[int]
    player_b_values: list[int]
    success: bool = False
    failure: bool = False
    aborted: bool = False
    proposal_a: list[int] | None = None
    proposal_b: list[int] | None = None


class DealOrNoDeal(DialogueGameMaster):
    '''
    This class implements a deal or no deal game in which players are negotiating
    about how to divide a set of items between each other. Players have different
    value functions for different types of items, and their goal is to optimize
    either their own score, or the sum/difference of the scores. They need to make
    a secret proposal at the end, and only if they are compatible will they receive
    any points.
    '''

    def __init__(self, game_name: str, game_path: str, experiment: dict, player_models: list[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

    def _on_setup(self, **game_instance):
        initial_prompt = self.experiment['initial_prompt']
        initial_prompt = initial_prompt \
            .replace('$N$', str(self.experiment['max_turns']))
        initial_prompt = initial_prompt \
            .replace('$ITEMS$', format_item_set(
                game_instance['item_counts'], game_instance['item_types'],
                game_instance['item_types_plural']
            ))
        player_a_initial_prompt = initial_prompt \
            .replace('$VALUE_FUNCTION$', format_value_function(
                game_instance['player_a_values'], game_instance['item_types']
            ))
        player_b_initial_prompt = initial_prompt \
            .replace('$VALUE_FUNCTION$', format_value_function(
                game_instance['player_b_values'], game_instance['item_types']
            ))
        self.player_a = DealOrNoSealPlayer(self.player_models[0])
        self.player_b = DealOrNoSealPlayer(self.player_models[1])
        self.add_player(self.player_a, initial_context=player_a_initial_prompt)
        self.add_player(self.player_b, initial_prompt=player_b_initial_prompt)
        # arguments in same order as above
        self.state = GameState(
            max_rounds=self.experiment['max_turns'],
            player_a_initial_prompt=player_b_initial_prompt,
            player_b_initial_prompt=player_b_initial_prompt,
            proposal_prompt_early=self.experiment['proposal_early'],
            proposal_prompt_timeout=self.experiment['proposal_timeout'],
            item_types=game_instance['item_types'],
            item_types_plural=game_instance['item_types_plural'],
            item_counts=game_instance['item_counts'],
            player_a_values=game_instance['player_a_values'],
            player_b_values=game_instance['player_b_values'],
        )
        # We use the stemmer to compare item names used by the models to be
        # somewhat lenient.
        self.stemmer = SnowballStemmer({
            'en': 'english',
            'de': 'german',
            'it': 'italian',
        }[self.experiment['language']])
