"""Bounded contact recovery during canonical SEARCH, disabled in cold replay.

Contact flags are geometric observations. This is not a force controller.
Commands never open a finger or change thumb opposition / abduction.
"""
import numpy as np
from botyard_two_stage_core import ACTUAL, POLICY, expand


class HoldRecovery:
    def __init__(self, target16, lower20, upper20, budget=.06, rate=.015):
        self.initial = np.asarray(target16, dtype=float).copy()
        self.command = self.initial.copy()
        self.upper = np.array([upper20[ACTUAL.index(n)] for n in POLICY])
        self.lower = np.array([lower20[ACTUAL.index(n)] for n in POLICY])
        if self.command.shape != (16,) or not np.isfinite(self.command).all():
            raise ValueError('Invalid holding target')
        if np.any(expand(self.command) < lower20) or np.any(expand(self.command) > upper20):
            raise ValueError('Holding target outside joint limits')
        self.flex = dict(ff=[1, 2], lf=[4, 5], mf=[7, 8], rf=[10, 11], th=[14, 15])
        self.maximum = self.initial.copy()
        # J2 and coupled distal J1 share a command; respect BOTH limits.
        for ids in self.flex.values():
            for i in ids:
                name = POLICY[i]
                limit = self.upper[i]
                if name.endswith('J2') and name[:2] != 'TH':
                    limit = min(limit, upper20[ACTUAL.index(name[:2]+'J1')])
                self.maximum[i] = min(limit, self.initial[i]+budget)
        self.rate = rate
        self.missing_time = dict.fromkeys(self.flex, 0.)
        self.contact_time = dict.fromkeys(self.flex, 0.)
        self.updates = dict.fromkeys(self.flex, 0)

    def update(self, row, dt):
        # Never squeeze against table/palm/cap support or after a physical drop.
        if row['contacts']['support'] or row['contacts']['capContacts'] or row['bottomZ'] < .03 or row['jointViolation'] > .01:
            return expand(self.command)
        q = np.array([row['q20'][ACTUAL.index(n)] for n in POLICY])
        for finger, ids in self.flex.items():
            absent = finger not in row['contacts']['fingers']
            self.missing_time[finger] = self.missing_time[finger]+dt if absent else 0.
            self.contact_time[finger] = 0. if absent else self.contact_time[finger]+dt
            # Geometric contact can persist while a finger has reached its
            # target. Restore a small error proxy only if BOTH flexion joints
            # are weak for sustained contact. This is not a normal-force test.
            weak = self.contact_time[finger]>=.2 and np.max(self.command[ids]-q[ids])<.005
            if self.missing_time[finger] < .06 and not weak:
                continue
            # Extra error capped at .025rad above ACTUAL state. If already
            # loaded, wait for dynamics rather than continuously winding up.
            limit = np.minimum(self.maximum[ids], q[ids]+(.01 if weak else .025))
            increment = np.maximum(0., np.minimum(self.rate*dt, limit-self.command[ids]))
            self.command[ids] += increment
            if np.any(increment > 0):
                self.updates[finger] += 1
        return expand(self.command)

    def report(self):
        return dict(initialTarget16=self.initial.tolist(), holdingTarget16=self.command.tolist(),
                    targetIncrement16=(self.command-self.initial).tolist(), maximumTarget16=self.maximum.tolist(),
                    updateCounts=self.updates.copy(), rateRadS=self.rate, missingContactDelaySeconds=.06,
                    weakContactDelaySeconds=.2, weakGapThresholdRad=.005, weakGapGoalRad=.01,
                    maximumPositiveErrorRad=.025, semantics='bounded geometric contact and position-error recovery; not measured force control')


def candidate_order(count, preferred):
    if len(set(preferred)) != len(preferred) or any(i < 0 or i >= count for i in preferred):
        raise ValueError('Preferred candidate indices must be unique and within trial cap')
    return list(preferred)+[i for i in range(count) if i not in preferred]
