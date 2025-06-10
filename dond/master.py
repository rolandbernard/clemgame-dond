
import re
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
    # e.g.: '5 hats, 1 egg, 2 books'
    return ', '.join([
        str(count) + ' ' + (single if count == 1 else plural)
        for count, single, plural in zip(counts, types, types_plural)
    ])


def format_value_function(values: list[int], types: list[str]) -> str:
    # e.g.: 'hat: 1, egg: 3, book: 2'
    return ', '.join([
        type + ': ' + str(value) for value, type in zip(values, types)
    ])


def find_matching_type(count: int, name: str, types: list[str], types_plural: list[str], stemmer: SnowballStemmer) -> str | None:
    # Check the expected plurality first.
    first = types if count == 1 else types_plural
    second = types_plural if count == 1 else types
    for words in [first, second]:
        idx = [i for i in range(len(types)) if words[i] == name]
        if len(idx) == 1:
            return types[idx[0]]
        # We guarantee that the singular and plural forms are unique.
        assert len(idx) == 0
    # If there is no match, try stemming it.
    for words in [first, second]:
        idx = [
            i for i in range(len(types))
            if stemmer.stem(words[i]) == stemmer.stem(name)
        ]
        if len(idx) == 1:
            return types[idx[0]]
        elif len(idx) > 1:
            return None  # Fail if there is ambiguity.
    return None


class DealOrNoSealPlayer(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        # Just some dummy sample responses.
        self._custom_responses = [
            'I will take nothing.', 'I will take everything.',
            'Lets divide things evenly!', 'Can you tell me what you value most?',
            '[Proposal:]', '[Proposal: 2 hats, 1 book]',
        ]

    def _custom_response(self, messages):
        word = self._custom_responses.pop(0)
        return f'{word}'


@dataclass
class GameState:
    mode: str
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
    success: bool = False   # Success is if a compromise is reached.
    failure: bool = False  # Failure is when the proposals were conflicting.
    aborted: bool = False   # Aborted means the game rules were broken.
    player_a_proposal: list[int] | None = None
    player_b_proposal: list[int] | None = None


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
        # Create initial prompt template for each player.
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
        # Add players.
        self.player_a = DealOrNoSealPlayer(self.player_models[0])
        self.player_b = DealOrNoSealPlayer(self.player_models[1])
        self.add_player(self.player_a, initial_context=player_a_initial_prompt)
        self.add_player(self.player_b, initial_prompt=player_b_initial_prompt)
        # Named arguments to avoid any order sensitivity.
        self.state = GameState(
            mode=self.experiment['mode'],
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

    def _does_game_proceed(self):
        return not (
            self.state.aborted or self.state.aborted
            or self.state.failure or self.state.success
        )

    def _parse_response(self, player: Player, response: str) -> str | list[int]:
        # Check if the message contains a proposal.
        match = re.search('\\[(.*?)\\]', response)
        if match:
            # This contains a proposal. Parse the specified syntax.
            match = re.search(
                '^Proposal:(()\\s*(\\d+)\\s+(\\w+)\\s*)((,|\\s+)\\s*(\\d+)\\s+(\\w+)\\s*)*,?$',
                match.groups()[0].strip(), re.IGNORECASE
            )
            if match is None:
                # The proposal submission syntax has not been followed.
                raise ParseError(
                    f'proposal must follow the stipulated syntax', response
                )
            counts = [0] * len(self.state.item_types)
            seen = [False] * len(self.state.item_types)
            # We allow items to be in any order.
            for _ in range(len(match.groups()) // 4):
                count = int(match.groups()[2])
                type = find_matching_type(
                    count, match.groups()[3], self.state.item_types,
                    self.state.item_types_plural, self.stemmer
                )
                if type is None:
                    # The proposal includes an item type that was not in the game.
                    raise ParseError(
                        f'proposal must include only valid item types', response
                    )
                index = self.state.item_types.index(type)
                if seen[index]:
                    # The proposal includes the same item type multiple times.
                    raise ParseError(
                        f'proposal must include every item type only once', response
                    )
                seen[index] = True
                counts[index] = count
            self.log_to_self('valid response', 'proposal')
            return counts
        else:
            self.log_to_self('valid response', 'continue')
            # If this is not a proposal, the message must not be parsed. The
            # players can communicate however they want.
            return response

    def _on_parse_error(self, error: ParseError):
        self.log_to_self('invalid format', 'abort game')
        self.state.aborted = True

    def _advance_game(self, player: Player, parsed_response: str | list[int]):
        # If this is a string, the message was not a proposal.
        if isinstance(parsed_response, str):
            if self.current_round == self.state.max_rounds \
                or self.state.player_a_proposal is not None \
                    or self.state.player_b_proposal is not None:
                # In these cases we expect a proposal. This is therefore a rule
                # violation.
                raise RuleViolationError(
                    f'player was instructed to make a proposal', parsed_response
                )
            self.log_to_self('valid message', parsed_response)
            if player == self.player_a:
                self.set_context_for(self.player_b, parsed_response)
            else:
                assert player == self.player_b
                # If this was the last allowed turn, we prompt the next player to
                # make their secret proposal.
                self.set_context_for(
                    self.player_a, parsed_response + '\n\n' +
                    self.state.proposal_prompt_timeout
                    if self.current_round == self.state.max_rounds - 1 else parsed_response
                )
        else:
            if any(
                proposed > count for count, proposed in zip(self.state.item_counts, parsed_response)
            ):
                # This was parsed correctly, but the user specified to many items.
                # We could count this as either a rule violation or a failed
                # proposal. I will count it as a rule violation.
                raise RuleViolationError(
                    'proposal includes more items than available',
                    str(parsed_response)
                )
            self.log_to_self('valid proposal', parsed_response)
            if player == self.player_a:
                self.state.player_a_proposal = parsed_response
                # Will be ignored if this is the end.
                self.set_context_for(
                    self.player_b, self.state.proposal_prompt_early
                )
            else:
                assert player == self.player_b
                self.state.player_b_proposal = parsed_response
                # Will be ignored if this is the end.
                self.set_context_for(
                    self.player_a, self.state.proposal_prompt_early
                )
            if self.state.player_a_proposal is not None \
                    and self.state.player_b_proposal is not None:
                if any(
                    # The sum of proposed items must be less than the available count.
                    cnta + cntb > count for count, cnta, cntb
                    in zip(self.state.item_counts, self.state.player_a_proposal, self.state.player_b_proposal)
                ):
                    # Record that the agreement failed.
                    self.log_to_self(
                        'failed agreement',
                        (self.state.player_a_proposal,
                         self.state.player_b_proposal)
                    )
                    self.state.failure = True
                else:
                    # Record that there was a successful agreement.
                    self.log_to_self(
                        'successful agreement',
                        (self.state.player_a_proposal,
                         self.state.player_b_proposal)
                    )
                    self.state.success = True

    def _on_game_error(self, error: GameError):
        self.log_to_self('error', error.reason)
        self.state.aborted = True

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))
