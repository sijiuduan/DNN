# https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil

import numpy as np

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def mv_mat3(tx,ty):
    mv = np.array([ 1, 0,tx],
                  [ 0, 1,ty],
                  [ 0, 0, 1])
    return mv

def sa_mat3(sx,sy):
    m = np.array([sx, 0, 0],
                 [ 0,sy, 0],
                 [ 0, 0, 1])
    return m


import math
def ro_mat3(a):
    m = [[ math.cos(a), math.sin(a),    0],
        [ -math.sin(a), math.cos(a),    0],
        [            0,           0,    1]
    ]
    return np.matrix(m)

# The matrix we actually want (note that it operates from the right):
# (1, 0, tx)   (1, 0, cx)   (sx, 0, 0)   ( cos a, sin a, 0)   (1, 0, -cx)
# (0, 1, ty) * (0, 1, cy) * ( 0,sy, 0) * (-sin a, cos a, 0) * (0, 1, -cy)
# (0, 0,  1)   (0, 0,  1)   ( 0, 0, 1)   (     0,     0, 1)   (0, 0,   1)

def rotation(angle,cx,cy):
    m1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
    r = angle/180.*math.pi
    sin = math.sin(r)
    cos = math.cos(r)
    m2 = np.array([[cos,sin,0],[-sin,cos,0],[0,0,1]])
    m3 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    m = np.dot(m3, np.dot(m2, m1))
    return m.reshape(1,9)[0][:8]

def scale(s,cx,cy):
    m1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
    m2 = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]])
    m3 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    m = np.dot(m3, np.dot(m2, m1))
    return m.reshape(1,9)[0][:8]

from PIL import Image

img = Image.open('a.png')
width, height = img.size

m = rotation(30,width/2,height/2)

img.transform((width, height), Image.PERSPECTIVE,m).save('b.png')

