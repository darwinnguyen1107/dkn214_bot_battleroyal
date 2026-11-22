"""Geometry and decision regression cases for Century Atlas."""
import math
import unittest
from array import array
from arena_tactics import Decisions
from code_ver10 import NavGrid, point_in_poly

def flat_grid():
    g=object.__new__(NavGrid)
    g.tminx=g.tminy=0.
    g.tstep=g.CELL
    g.tnx=g.tny=g.gnx=g.gny=8
    g.terrain=array("h",[0])*64
    g.wblock=bytearray(64)
    g.top=array("f",[-1e9])*64
    g.spans={};g.learned={};g.risk_cache={};g.rebuild_context()
    return g

class Geometry(unittest.TestCase):
    def test_concave(self):
        ring=[0,0,6,0,6,2,2,2,2,6,0,6]
        self.assertTrue(point_in_poly(1,5,ring))
        self.assertFalse(point_in_poly(5,5,ring))
        self.assertFalse(point_in_poly(0,0,[]))

    def test_walk(self):
        g=flat_grid()
        self.assertTrue(g.walk_clear((1,1),(16,1)))
        g.learned[g.cell_of(8,1)]=100
        self.assertFalse(g.walk_clear((1,1),(16,1)))
        self.assertFalse(g.walk_clear((1,1),(-4,1)))

    def test_bridge(self):
        g=flat_grid();g.spans[g.cell_of(8,1)]=[(10,14)]
        self.assertTrue(g.shot_clear((1,1,5),(17,1,5)))
        self.assertFalse(g.shot_clear((1,1,12),(17,1,12)))
        self.assertIn(14,g.floors(g.cell_of(8,1)))

    def test_lob(self):
        bot=object.__new__(Decisions);bot.nav=None
        yaw,pitch,t=bot.lob(0,0,5,(70,0,5),(0,0,0),72,22)
        self.assertAlmostEqual(yaw,0)
        self.assertAlmostEqual(72*math.cos(pitch)*t,70)
        self.assertAlmostEqual(72*math.sin(pitch)*t-11*t*t,0,places=4)
        self.assertIsNone(bot.lob(0,0,5,(600,0,5),(0,0,0),72,22))

    def test_dodge(self):
        bot=object.__new__(Decisions);bot.nav=None;bot.impacts=[]
        me={"pos":[0,0,0],"vel":[0,0,0]}
        self.assertIsNone(bot.dodge(me,[],(0,0),50))
        shot={"kind":"gun","pos":[-20,0,4],"vel":[40,0,0]}
        dirs=lambda _: [(0,1)]
        self.assertEqual(bot.dodge(me,[shot],(0,0),50,dirs),(0,1))
        self.assertIsNone(bot.dodge(me,[shot],(0,0),50,lambda _:[]))
        self.assertTrue(bot.jump_dodge)

if __name__=="__main__":
    unittest.main()
