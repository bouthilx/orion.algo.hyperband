# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.hyperband.hyperband -- TODO
=================================================

.. module:: hyperband
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
import copy
import logging
import hashlib

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Fidelity


logger = logging.getLogger(__name__)


REGISTRATION_ERROR = """
Bad fidelity level {fidelity}. Should be in {budgets}.
Params: {params}
"""

SPACE_ERROR = """
Hyperband cannot be used if space does contain a fidelity dimension.
For more information on the configuration and usage of Hyperband, see
https://orion.readthedocs.io/en/develop/user/algorithms.html#hyperband
"""

BUDGET_ERROR = """
Cannot build budgets below max_resources;
(max: {}) - (min: {}) > (num_rungs: {})
"""


def compute_budgets(min_resources, max_resources, reduction_factor, num_rungs):
    """Compute the budgets used for Hyperband"""
    budgets = numpy.logspace(
        numpy.log(min_resources) / numpy.log(reduction_factor),
        numpy.log(max_resources) / numpy.log(reduction_factor),
        num_rungs, base=reduction_factor).astype(int)

    for i in range(num_rungs - 1):
        if budgets[i] >= budgets[i + 1]:
            budgets[i + 1] = budgets[i] + 1

    if budgets[-1] > max_resources:
        raise ValueError(BUDGET_ERROR.format(min_resources, max_resources, num_rungs))

    return list(budgets)


def compute_rung_sizes(reduction_factor, num_rungs):
    return [reduction_factor**i for i in range(num_rungs)][::-1]


class Hyperband(BaseAlgorithm):
    """Hyperband
    
    `Hyperparameter optimization [formulated] as a pure-exploration non-stochastic
    infinite-armed bandit problem where a predefined resource like iterations, data samples, or features
    is allocated to randomly sampled configurations.``

    For more information on the algorithm, see original paper at http://jmlr.org/papers/v18/16-558.html.

    Li, Lisha et al. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
    Journal of Machine Learning Research, 18:1–52, 2018.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    num_rungs: int, optional
        Number of rungs for the largest bracket. If not defined, it will be equal to (base + 1) of
        the fidelity dimension. In the original paper,
        num_rungs == log(fidelity.high/fidelity.low) / log(fidelity.base) + 1.
        Default: log(fidelity.high/fidelity.low) / log(fidelity.base) + 1
    num_brackets: int
        Using a grace period that is too small may bias Hyperband too strongly towards
        fast converging trials that do not lead to best results at convergence (stagglers). To
        overcome this, you can increase the number of brackets, which increases the amount of
        resource required for optimisation but decreases the bias towards stragglers.
        Default: 1

    """

    def __init__(self, space, seed=None, num_rungs=None, num_brackets=1):
        super(Hyperband, self).__init__(
            space, seed=seed, num_rungs=num_rungs, num_brackets=num_brackets)

        self.trial_info = {}  # Stores Trial -> Bracket

        try:
            fidelity_index = self.fidelity_index
        except IndexError:
            raise RuntimeError(SPACE_ERROR)

        fidelity_dim = space.values()[fidelity_index]

        min_resources = fidelity_dim.low
        max_resources = fidelity_dim.high
        reduction_factor = fidelity_dim.base

        if reduction_factor < 2:
            raise AttributeError("Reduction factor for Hyperband needs to be at least 2.")

        if num_rungs is None:
            num_rungs = int(numpy.log(max_resources / min_resources) /
                            numpy.log(reduction_factor) + 1)

        self.num_rungs = num_rungs

        budgets = compute_budgets(min_resources, max_resources, reduction_factor, num_rungs)

        # Tracks state for new trial add
        self.brackets = [
            Bracket(self, reduction_factor, budgets[bracket_index:])
            for bracket_index in range(num_brackets)
        ]

    def sample(self, num):
        return list(self.space.sample(num, seed=tuple(self.rng.randint(0, 1000000, size=3))))

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.rng = numpy.random.RandomState(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {'rng_state': self.rng.get_state()}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.seed_rng(0)
        self.rng.set_state(state_dict['rng_state'])

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        Sample new points until first rung is filled. Afterwards 
        waits for all trials to be completed before promoting trials
        to the next rung.

        Parameters
        ----------
        num: int, optional
            Number of points to suggest. Defaults to 1.

        Returns
        -------
        list of points or None
            A list of lists representing points suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        """
        if num > 1:
            raise ValueError("Hyperband should suggest only one point.")

        for bracket in self.brackets:
            if not bracket.is_filled:
                return [tuple(bracket.sample())]

        # All brackets are filled

        for bracket in self.brackets:
            if bracket.is_ready():
                return [tuple(bracket.promote())]

        # Either all brackets are done or none are ready and algo needs to wait for some trials to
        # complete
        return None

    def get_id(self, point):
        """Compute a unique hash for a point based on params, but not fidelity level."""
        _point = list(point)
        non_fidelity_dims = _point[0:self.fidelity_index]
        non_fidelity_dims.extend(_point[self.fidelity_index + 1:])

        return hashlib.md5(str(non_fidelity_dims).encode('utf-8')).hexdigest()

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """
        for point, result in zip(points, results):

            _id = self.get_id(point)
            bracket = self.trial_info.get(_id)

            if not bracket:
                fidelity = point[self.fidelity_index]
                brackets = [bracket for bracket in self.brackets
                            if bracket.rungs[0][0] == fidelity]
                if not brackets:
                    raise ValueError(
                        "No bracket found for point {0} with fidelity {1}".format(_id, fidelity))
                bracket = brackets[0]

            try:
                bracket.register(point, result['objective'])
            except IndexError:
                logger.warning('Point registered to wrong bracket. This is likely due '
                               'to a corrupted database, where trials of different fidelity '
                               'have a wrong timestamps.')
                continue

            if _id not in self.trial_info:
                self.trial_info[_id] = bracket

    @property
    def is_done(self):
        """Return True, if all brackets reached their maximum resources."""
        return all(bracket.is_done for bracket in self.brackets)

    @property
    def fidelity_index(self):
        """Compute the index of the point when fidelity is."""
        def _is_fidelity(dim):
            return (isinstance(dim, Fidelity) or
                    (hasattr(dim, 'original_dimension') and
                     isinstance(dim.original_dimension, Fidelity)))

        return [i for i, dim in enumerate(self.space.values()) if _is_fidelity(dim)][0]


class Bracket():
    """Bracket of rungs for the algorithm Hyperband.

    Parameters
    ----------
    hyperband: `Hyperband` algorithm
        The hyperband algorithm object which this bracket will be part of.
    reduction_factor: int
        The factor by which Hyperband promotes trials. If the reduction factor is 4,
        it means the number of trials from one fidelity level to the next one is roughly
        divided by 4, and each fidelity level has 4 times more resources than the prior one.
    budgets: list of int
        Budgets used for each rung

    """

    def __init__(self, hyperband, reduction_factor, budgets):
        self.hyperband = hyperband
        self.reduction_factor = reduction_factor
        self.rungs = [(budget, dict()) for budget in budgets]

        logger.debug('Bracket budgets: %s', str([rung[0] for rung in self.rungs]))

        # points = hyperband.sample(compute_rung_sizes(reduction_factor, len(budgets))[0])
        # for point in points:
        #     self.register(point, None)

    @property
    def is_filled(self):
        """Return True if last rung with trials is filled"""
        return self.has_rung_filled(0)

    def sample(self):
        """Sample a new trial with lowest fidelity"""
        point = list(self.hyperband.sample(1)[0])
        point[self.hyperband.fidelity_index] = self.rungs[0][0]
        return point

    def register(self, point, objective):
        """Register a point in the corresponding rung"""
        fidelity = point[self.hyperband.fidelity_index]
        rungs = [rung for budget, rung in self.rungs if budget == fidelity]
        if not rungs:
            budgets = [budget for budget, rung in self.rungs]
            raise IndexError(REGISTRATION_ERROR.format(fidelity=fidelity, budgets=budgets,
                                                       params=point))

        rungs[0][self.hyperband.get_id(point)] = (objective, point)

    def get_candidate(self, rung_id):
        """Get a candidate for promotion"""
        if self.has_rung_filled(rung_id + 1):
            return None

        _, rung = self.rungs[rung_id]
        next_rung = self.rungs[rung_id + 1][1]

        rung = list(sorted((objective, point) for objective, point in rung.values()))

        for trial in rung:
            objective, point = trial
            assert objective is not None
            _id = self.hyperband.get_id(point)
            if _id not in next_rung:
                return point

        return None

    @property
    def is_done(self):
        """Return True, if the last rung is filled."""
        return len(self.rungs[-1][1])

    def has_rung_filled(self, rung_id):
        """Return True, if the rung[rung_id] is filled."""
        n_trials = len(self.rungs[rung_id][1])
        return n_trials >= compute_rung_sizes(self.reduction_factor, len(self.rungs))[rung_id]

    def is_ready(self, rung_id=None):
        if rung_id is not None:
            return (
                self.has_rung_filled(rung_id) and
                all(objective is not None for objective, _ in self.rungs[rung_id][1].values()))

        is_ready = False
        for rung_id in range(len(self.rungs)):
            if self.has_rung_filled(rung_id):
                is_ready = self.is_ready(rung_id)
            else:
                break

        return is_ready

    def promote(self):
        """Promote the first candidate that is found and return it

        The rungs are iterated over in reversed order, so that high rungs
        are prioritised for promotions. When a candidate is promoted, the loop is broken and
        the method returns the promoted point.

        .. note ::

            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.

        """
        if self.is_done:
            return None

        for rung_id in range(len(self.rungs)):
            if self.has_rung_filled(rung_id + 1):
                continue

            if not self.is_ready(rung_id):
                return None

            candidate = self.get_candidate(rung_id)
            if candidate:

                # pylint: disable=logging-format-interpolation
                logger.debug(
                    'Promoting {point} from rung {past_rung} with fidelity {past_fidelity} to '
                    'rung {new_rung} with fidelity {new_fidelity}'.format(
                        point=candidate, past_rung=rung_id,
                        past_fidelity=candidate[self.hyperband.fidelity_index],
                        new_rung=rung_id + 1, new_fidelity=self.rungs[rung_id + 1][0]))

                candidate = list(copy.deepcopy(candidate))
                candidate[self.hyperband.fidelity_index] = self.rungs[rung_id + 1][0]

                return tuple(candidate)

        return None

    def __repr__(self):
        """Return representation of bracket with fidelity levels"""
        return 'Bracket({})'.format([rung[0] for rung in self.rungs])
