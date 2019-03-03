"""provides vector align function"""

import numpy as np

def ell_valign(vec_v, vec_x):

    """ produces mapping matrix that aligns vector X to vector V"""
    #check for dimension, raise error
    try:
        vec_v_left_rotation, vec_v_stretch, vec_v_right_rotation = np.linalg.svd(vec_v, full_matrices=True)
        vec_x_left_rotation, vec_x_stretch, vec_x_right_rotation = np.linalg.svd(vec_x, full_matrices=True)
        #has to be full SVD

        #sketch way of dealing with scalars
        if vec_v_left_rotation.shape[0] == 1 and vec_v_left_rotation.shape[1] == 1:
            vec_v_left_rotation = vec_v_left_rotation.item()
        if vec_x_left_rotation.shape[0] == 1 and vec_x_left_rotation.shape[1] == 1:
            vec_x_left_rotationtranspose = vec_x_left_rotation.item()
        else:
            vec_x_left_rotationtranspose = vec_x_left_rotation.T
        if vec_v_right_rotation.shape[0] == 1 and vec_v_right_rotation.shape[1] == 1:
            vec_v_right_rotation = vec_v_right_rotation.item()
        if vec_x_right_rotation.shape[0] == 1 and vec_x_right_rotation.shape[1] == 1:
            vec_x_right_rotationtranspose = vec_x_right_rotation.item()
        else:
            vec_x_right_rotationtranspose = vec_x_right_rotation.T
#
        mapping_matrix_s = vec_v_left_rotation*vec_v_right_rotation*vec_x_right_rotationtranspose*vec_x_left_rotationtranspose
    except Exception as e:
        print(e)
        raise ValueError("invalid vector inputs for ell_valign")

    return mapping_matrix_s



def main():
    """stand alone testing"""
    vec_xl0 = np.matrix('1; 0')
    vec_l = np.matrix('2.4016; 9.3249')

    print(vec_xl0)
    print(vec_l)
    parallelizer_s = ell_valign(vec_xl0, vec_l)
    print(parallelizer_s)

if __name__ == "__main__":
    main()
