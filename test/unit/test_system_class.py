import numpy as np
from src.ell_input_data import EllSystem
from pytest import approx

#System initiailization defaults
#check matrix ordering
#check sqrt accuracy
#check time value/segmentation check
#check system update



@pytest.mark.parametrize('system_description', system_list)
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


