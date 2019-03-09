import numpy as np
from src.ell_input_data import EllSystem
import pytest
from pytest import approx

import src.reach_disturbance

#System initiailization defaults
#check matrix ordering
#check sqrt accuracy
#check time value/segmentation check
#check system update




def test_switch():
    sys = EllSystem(t_end=4)

    expectedA = np.reshape(np.array([[0.000], [-10.00],
                              [2.0000], [-8.000],
                              ]),(2,2))

    cmp_version1 = np.matrix('0.000 -10.000; 2.000 -8.000')
    cmp_version2 = np.matrix([[0.000,-10.000],[2.000,-8.000]])


    assert sys.A == approx(expectedA)
    assert sys.A == approx(cmp_version1)
    assert sys.A == approx(cmp_version2)


    sys.switch_system()

    expectedA = np.reshape(np.array([[0, -10],
                      [1, -2],
                      ]),(2,2))

    assert sys.A == approx(expectedA)

@pytest.mark.parametrize('propagation_time', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,2])
def test_time_parameters(propagation_time):

    system_description = {
        'A': np.matrix('0.000 -10.000; 2.000 -8.000'),
        'B': np.matrix('10.000 0.000; 0.000 2.000'),
        'P': np.matrix('1.000 0.000; 0.000 1.000'),
        'L': np.matrix('1.000 0.000; 0.000 1.000'),
        'X0': np.matrix('1.000 0.000; 0.000 1.000'),
        'XC': np.matrix('0.000; 0.000'),
        'BC': np.matrix('0.000; 0.000')
    }

    sys = EllSystem(system_description=system_description, t_end=propagation_time)  # SETUP DYNAMIC SYSTEM DESCRIPTION HERE
    print("starting center traj calculation")
    center_trajectory = src.reach_disturbance.reach_center(sys, y0=[], evolve=False, debug=False)
    print("done center traj calculation")

    EA_reach_set = np.empty((sys.n * sys.n, sys.len_prop, sys.num_search))
    IA_reach_set = np.empty((sys.n * sys.n, sys.len_prop, sys.num_search))
    time_tube = []  # TIME_STAMP RECORD TO GUARANTEE CORRECT TIME SCALING OF THE GUI

    for i in range(sys.num_search):
        # sys.L0 = sys.L[i].T
        sys.L0 = sys.L[i].T

        print("sys.L0: " + str(sys.L0))
        # sys.xl0 = np.asmatrix(np.linalg.multi_dot([sys.M,sys.L0])).T
        sys.xl0 = (np.linalg.multi_dot([sys.M, sys.L0]))
        print("sys.xl0: " + str(sys.xl0))

        IA_tube, IA_time_tube = src.reach_disturbance.IA_reach_per_search(sys)

        #assert time_tube[-1] == approx(5.0,rel=sys.dt)

        EA_tube, EA_time_tube = src.reach_disturbance.EA_reach_per_search(sys)

        #assert time_tube[-1] == approx(5.0,rel=sys.dt)

        #EA_reach_set[:, :, i] = EA_tube  # time_series x vectorized shape matrix x search direction
        #IA_reach_set[:, :, i] = IA_tube

    assert IA_time_tube[0] ==approx(0.0)
    assert EA_time_tube[0] == approx(0.0)
    assert sys.time_grid[-1] == approx(propagation_time)
    assert sys.time_grid[-1] == approx(EA_time_tube[-1])
    assert IA_time_tube[-1] == approx(propagation_time)
    assert EA_time_tube[-1] == approx(propagation_time)
    assert len(center_trajectory[0,:]) == sys.time_grid_size







