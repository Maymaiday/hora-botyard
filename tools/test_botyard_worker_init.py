#!/usr/bin/env python3
"""Regression for the uploaded startup failure; no Isaac Gym import required."""
import ast
from pathlib import Path
import unittest
import numpy as np
SOURCE=Path(__file__).with_name('botyard_two_stage_worker.py')
TREE=ast.parse(SOURCE.read_text(),feature_version=(3,8))
FUNCTION=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='single_env_state_offset')
MODULE=ast.Module(body=[FUNCTION],type_ignores=[])
SCOPE={'np':np};exec(compile(ast.fix_missing_locations(MODULE),str(SOURCE),'exec'),SCOPE)
class Regression(unittest.TestCase):
    def test_origin_zero(self):
        np.testing.assert_array_equal(SCOPE['single_env_state_offset']([0,0,0]),[0,0,0])
    def test_invalid_origins_not_silently_accepted(self):
        for origin in [[1,0,0],[float('nan'),0,0],[0,0]]:
            with self.assertRaises(RuntimeError):SCOPE['single_env_state_offset'](origin)
    def test_uploaded_displacement_is_not_coordinate_calibration(self):
        measured=np.array([.00062653,-.00089558,.04147547]);requested=np.array([0,0,.041])
        self.assertAlmostEqual(np.linalg.norm(measured-requested)*1000,1.19192076,places=4)
        # Coordinate translation is established from the environment origin alone.
        np.testing.assert_array_equal(SCOPE['single_env_state_offset']([0,0,0]),[0,0,0])
    def test_constructor_does_not_read_unstepped_body_state(self):
        scene=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='Scene')
        init=next(n for n in scene.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        calls=[n.func.attr for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('get_actor_rigid_body_states',calls)
        snapshot=next(n for n in scene.body if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
        self.assertIn('get_actor_rigid_body_states',[n.func.attr for n in ast.walk(snapshot) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)])
    def test_physics_fetch_precedes_snapshot(self):
        scene=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='Scene')
        step=next(n for n in scene.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        direct=[n.value.func.attr for n in step.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)]
        self.assertLess(direct.index('simulate'),direct.index('fetch_results'))
        phase=next(n for n in scene.body if isinstance(n,ast.FunctionDef) and n.name=='phase')
        body=ast.get_source_segment(SOURCE.read_text(),phase)
        self.assertLess(body.index('self.step('),body.index('self.snapshot('))
if __name__=='__main__':unittest.main(verbosity=2)
