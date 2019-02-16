import numpy as np
from src.ell_input_data import System

#System initiailization defaults
#check matrix ordering
#check sqrt accuracy
#check time value/segmentation check


def test_switch():
    sys = System(t_end=4)

    expectedA = np.reshape(np.array([[0.000], [-10.00],
                              [2.0000], [-8.000],
                              ]),(2,2))

    np.testing.assert_almost_equal(sys.A, expectedA, decimal=7, err_msg='', verbose=True)


    sys.switch_system()

    expectedA = np.reshape(np.array([[0, -10],
                      [1, -2],
                      ]),(2,2))
    np.testing.assert_almost_equal(sys.A, expectedA, decimal=7, err_msg='', verbose=True)




