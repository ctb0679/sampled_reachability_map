#!/usr/bin/env python3

import sympy as sp
from sympy import lambdify
import os
import getpass


theta1,theta2,theta3,theta4,theta5,theta6 = sp.symbols('theta1 theta2 theta3 theta4 theta5 theta6')

def find_file_directory(file_name, start_directory):
    for root, dirs, files in os.walk(start_directory):
        if file_name in files:
            return root
    return None

username = getpass.getuser()
symb_dir = find_file_directory('symb_jacobian.txt', ("/home/"+username+"/inspection_ws/src/"))


with open((symb_dir + "/symb_jacobian.txt"), "r") as inf:
    J = sp.Matrix(sp.sympify(inf.read()))

with open((symb_dir + "/symb_transform.txt"), "r") as inf:
    A = sp.Matrix(sp.sympify(inf.read()))

with open((symb_dir + "/symb_transform_matrix.txt"), "r") as inf:
    T = sp.Matrix(sp.sympify(inf.read()))

A_lamb = (lambdify((theta1,theta2,theta3,theta4,theta5,theta6), A, 'numpy'))
J_lamb = (lambdify((theta1,theta2,theta3,theta4,theta5,theta6), J, 'numpy'))
T_lamb = (lambdify((theta1,theta2,theta3,theta4,theta5,theta6), T, 'numpy'))
