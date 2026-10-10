"""Rigid-contact classification for a cylinder aligned with its local Z axis."""
import numpy as np

def xyz(v):
    if getattr(v, 'dtype', None) is not None and v.dtype.names:
        return np.array([v['x'], v['y'], v['z']], dtype=float)
    if hasattr(v, 'x'):
        return np.array([v.x, v.y, v.z], dtype=float)
    return np.asarray(v, dtype=float).reshape(3)

def classify_contacts(contacts, object_handle, distal_handles, support_handles,
                      radius, length, cap_margin=0.006, radial_tolerance=0.004):
    """Use object-local contact positions; never infer side contacts from body IDs.

    All lengths are in meters, including actor scale. Requires Isaac Gym's
    named RigidContact fields. Fails explicitly if the API differs.
    """
    side, support, cap = set(), False, False
    for c in contacts:
        names = c.dtype.names
        position0 = 'localPos0' if names and 'localPos0' in names else 'local_pos0'
        position1 = 'localPos1' if names and 'localPos1' in names else 'local_pos1'
        if not names or not {'body0', 'body1', position0, position1}.issubset(names):
            raise RuntimeError('RigidContact has no named localPos0/localPos1 fields: %r' % (names,))
        b0,b1=int(c['body0']),int(c['body1'])
        if b0 == object_handle:
            other,p=b1,xyz(c[position0])
        elif b1 == object_handle:
            other,p=b0,xyz(c[position1])
        else:
            continue
        # Friction acts at feature + offset, when provided by this Gym build.
        offset = ('offset0' if b0 == object_handle else 'offset1')
        if offset in names:
            p = p + xyz(c[offset])
        if other in support_handles:
            support=True
        if other in distal_handles or other in support_handles:
            # Exclude contacts near either cap, even if also near its rim.
            if abs(p[2]) >= length / 2 - cap_margin:
                cap=True
            elif other in distal_handles and abs(np.linalg.norm(p[:2])-radius) <= radial_tolerance:
                side.add(other)
    return side,support,cap
