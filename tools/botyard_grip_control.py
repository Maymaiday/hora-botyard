"""Bounded contact closure and named collision masks; no simulator dependency.

Collision pairs transcribed from wangrp1997/tendon-spin b41630f4,
assets/grasp/scene.xml. This controller is a new Isaac Gym adaptation,
not a reproduction of their Isaac Lab controller or a hardware calibration.
"""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from botyard_two_stage_core import ACTUAL, POLICY, FINGERS, expand

def forefinger_target(q20,target16,lower20,upper20,margin):
    q=np.asarray(q20,dtype=float);target=np.asarray(target16,dtype=float).copy()
    if q.shape!=(20,) or target.shape!=(16,) or not np.isfinite(q).all() or not np.isfinite(target).all() or not 0<margin<=.06:
        raise ValueError('Invalid forefinger preload inputs')
    target[[1,2]]=np.maximum(target[[1,2]],q[[1,2]]+margin)
    expanded=expand(target)
    if np.any(expanded<lower20) or np.any(expanded>upper20):raise ValueError('Forefinger preload proposal exceeds joint limits')
    return target


def collision_masks(names, pairs):
    if len(pairs) > 30:
        raise ValueError('Too many pairs for positive signed 32-bit collision masks')
    masks = dict.fromkeys(names, 0)
    requested = set()
    for i, (a, b) in enumerate(pairs):
        if a == b or a not in masks or b not in masks:
            raise ValueError('Unresolved collision pair: '+str((a, b)))
        pair = frozenset((a, b))
        if pair in requested:
            raise ValueError('Duplicate collision pair')
        requested.add(pair)
        masks[a] |= 1 << i
        masks[b] |= 1 << i
    actual = {frozenset((a, b)) for a, b in itertools.combinations(names, 2)
              if masks[a] & masks[b]}
    if actual != requested:
        raise ValueError('Collision masks exclude unintended pairs')
    return masks


def tracking_check(rows, tolerance=.08, speed_limit=.5):
    if not rows:
        return dict(passed=False, reasons=['no_tracking_samples'])
    end = rows[-1]['simTime']
    tail = [r for r in rows if r['simTime'] >= end-.10]
    errors = np.array([np.abs(np.asarray(r['target20'])-r['q20']) for r in tail])
    speeds = np.abs(np.array([r['velocity20'] for r in tail]))
    bad = [dict(joint=n, maxErrorRad=float(errors[:, i].max()),
                maxSpeedRadS=float(speeds[:, i].max())) for i, n in enumerate(ACTUAL)
           if errors[:, i].max() > tolerance or speeds[:, i].max() > speed_limit]
    return dict(passed=not bad, toleranceRad=tolerance, speedLimitRadS=speed_limit,
                reasons=['empty_hand_target_tracking_failed'] if bad else [], joints=bad)


class ContactCloser:
    """Advance flexion only; latch sustained contact and apply bounded preload.

    Contact flags are geometric side contacts, NOT calibrated force readings.
    Thumb opposition J4/J3 remains at its candidate target. Four distal joints
    still use equal position targets in Gym; this is not a mechanical tendon.
    """
    def __init__(self, target16, lower20, upper20, rate=.12, extra=.35, preload=.025):
        self.command = np.asarray(target16, dtype=float).copy()
        self.lower = np.array([lower20[ACTUAL.index(n)] for n in POLICY])
        self.upper = np.array([upper20[ACTUAL.index(n)] for n in POLICY])
        self.maximum = self.command.copy()
        self.flex = dict(ff=[1, 2], lf=[4, 5], mf=[7, 8], rf=[10, 11], th=[14, 15])
        for ids in self.flex.values():
            self.maximum[ids] = np.minimum(self.upper[ids], self.command[ids]+extra)
        self.rate, self.preload = rate, preload
        self.contact_time = dict.fromkeys(FINGERS, 0.)
        self.latched = dict.fromkeys(FINGERS, False)
        self.contact_q = {}

    def update(self, q20, contacts, dt):
        q16 = np.array([q20[ACTUAL.index(n)] for n in POLICY])
        for finger, ids in self.flex.items():
            touching = finger in contacts
            self.contact_time[finger] = self.contact_time[finger]+dt if touching else 0.
            if not touching:
                self.latched[finger] = False
                self.command[ids] = np.minimum(self.maximum[ids], self.command[ids]+self.rate*dt)
            elif self.contact_time[finger] >= .06 and not self.latched[finger]:
                self.latched[finger] = True
                self.contact_q[finger] = q16[ids].tolist()
                # Actual contact angle + small elastic error, not full unused
                # closure target. Never exceed the declared closure budget.
                self.command[ids] = np.clip(q16[ids]+self.preload,
                                            self.lower[ids], self.maximum[ids])
        return expand(self.command)

    def report(self):
        return dict(holdingTarget16=self.command.tolist(), maximumTarget16=self.maximum.tolist(),
                    latchedFingers=[f for f in FINGERS if self.latched[f]],
                    contactFlexion16ByFinger=self.contact_q, rateRadS=self.rate,
                    preloadRad=self.preload, semantics='geometric contact latch; not force regulation')


def closed_grip_check(rows, scale, settled=False):
    if not rows:
        return dict(passed=False, reasons=['no_closure_samples'])
    tail = [r for r in rows if r['simTime'] >= rows[-1]['simTime']-.3]
    fractions = {f: sum(f in r['contacts']['fingers'] for r in tail)/len(tail) for f in FINGERS}
    reasons = []
    if any(v < .9 for v in fractions.values()): reasons.append('five_finger_side_contact_not_established')
    if fractions['th'] < .9: reasons.append('thumb_side_contact_not_established')
    if any(set(r['contacts']['support'])-{'ground'} for r in tail): reasons.append('palm_or_other_support_during_close')
    if any(r['contacts']['capContacts'] for r in tail): reasons.append('finger_cap_contact_during_close')
    heights = [z for r in tail for z in r['contacts']['sideContactZByFinger'].values()]
    band = float(np.mean(heights)) if heights else None
    if band is None or not 0 <= band <= .02*scale: reasons.append('contact_band_not_above_center')
    linear=np.array([np.linalg.norm(r['objectVelocity6'][:3]) for r in tail])
    angular=np.array([np.linalg.norm(r['objectVelocity6'][3:]) for r in tail])
    within=(linear<=.03)&(angular<=.3)
    motion=dict(velocityWithinFraction=float(within.mean()),maxLinearSpeed=float(linear.max()),
                maxAngularSpeed=float(angular.max()),linearLimit=.03,angularLimit=.3,
                rule='95% within limits + bounded peaks + pose stability' if settled else 'every sample within limits')
    if settled:
        # Declared revised pre-lift gate. Keep strict old score separately.
        positions=np.array([r['object7'][:3] for r in tail])
        position_change=float(np.linalg.norm(positions-positions[0],axis=1).max())
        rotations=Rotation.from_quat([r['object7'][3:] for r in tail])
        angle_change=float((rotations[0].inv()*rotations).magnitude().max())
        motion.update(maxPositionChangeM=position_change,maxOrientationChangeDeg=float(np.degrees(angle_change)),
                      positionLimitM=.001,orientationLimitDeg=1.,peakLinearLimit=.06,peakAngularLimit=.6)
        if within.mean()<.95 or linear.max()>.06 or angular.max()>.6 or position_change>.001 or angle_change>np.deg2rad(1):
            reasons.append('object_not_settled_after_close')
    elif not within.all():reasons.append('object_not_settled_after_close')
    return dict(passed=not reasons, reasons=reasons, tailSeconds=.3,
                sideContactFractions=fractions, meanContactBandZ=band,motion=motion)
