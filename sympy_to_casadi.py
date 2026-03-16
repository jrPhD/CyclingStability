import platform

import sympy as sm
import sympy.physics.mechanics as me
import numpy as np


mapping_sympy2casadi = {
    "Abs": "cas.fabs",
    "sin": "cas.sin",
    "cos": "cas.cos",
    "tan": "cas.tan",
    "asin": "cas.asin",
    "acos": "cas.acos",
    "atan": "cas.atan",
    "sinh": "cas.sinh",
    "cosh": "cas.cosh",
    "tanh": "cas.tanh",
    "asinh": "cas.asinh",
    "acosh": "cas.acosh",
    "atanh": "cas.atanh",
    "exp": "cas.exp",
    "log": "cas.log",
    "sqrt": "cas.sqrt",
}

variable_list = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "q8",
    "u1",
    "u2",
    "u3",
    "u4",
    "u5",
    "u6",
    "u7",
    "u8",
    "steer_torque",
    "disturbance",
]

mapping_derivatives = {
    "Derivative(q1, t)": "u1",
    "Derivative(q2, t)": "u2",
    "Derivative(q3, t)": "u3",
    "Derivative(q4, t)": "u4",
    "Derivative(q5, t)": "u5",
    "Derivative(q6, t)": "u6",
    "Derivative(q7, t)": "u7",
    "Derivative(q8, t)": "u8"}


def generate_model_file(
    file_name: str,
    list_function_names: list[str],
    sympy_expr: list[sm.matrices.dense.MutableDenseMatrix],
    variable_list: list[str],
    constants: dict[str, float],
):

    def write_expr_in_txt_file(
        function_name: str,
        matrix: sm.matrices.dense.MutableDenseMatrix,
    ):

        n, m = np.shape(matrix)

        f.write(f"n={n}")
        f.write("\n")

        f.write(f"m={m}")
        f.write("\n")

        for i in range(n):
            for j in range(m):
                print('writting element', n, m)

                expr = f"{matrix[i,j]}"
                expr = expr.replace("bike_v1_0_", "")

                for var in variable_list:
                    expr = expr.replace(f"{var}(t)", f"{var}")

                for key in mapping_sympy2casadi.keys():
                    expr = expr.replace(f"{key}(", f"{mapping_sympy2casadi[key]}(")
                    
                for key in mapping_derivatives.keys():
                        expr = expr.replace(f"{key}", f"{mapping_derivatives[key]}")

                f.write(f"{name}_{i}_{j}={expr}")

                f.write("\n")
                print(f"{name}_{i}_{j} declared with success")

        f.write(f"{name}=cas.SX(n,m)")
        f.write("\n")

        for i in range(n):
            for j in range(m):
                f.write(f"{name}[{i},{j}]={name}_{i}_{j}")
                f.write("\n")

    if platform.system() == "Windows":
        full_file_name = f"model_files\{file_name}.py"
    else:
        full_file_name = f"model_files/{file_name}.py"

    with open(full_file_name, "w") as f:

        f.write("import casadi as cas")
        f.write("\n")
        f.write("\n")

        # Declare casadi variables and constants
        f.write("#Declare variables")
        for var in variable_list:

            f.write("\n")
            f.write(f"{var} = cas.SX.sym('{var}', 1, 1)")
        f.write("\n")
        f.write("\n")

        f.write("list_variables = [")
        f.write("\n")

        for var in variable_list:
            f.write(f"{var},")
            f.write("\n")
        f.write("]")
        f.write("\n")
        f.write("\n")

        f.write("#Declare contants")
        for cons in constants.keys():
            f.write("\n")
            f.write(f"{cons} = cas.SX.sym('{cons}', 1, 1)")
        f.write("\n")
        f.write("\n")
        print("Variables and constants declared with success")

        f.write("list_constants = [")
        f.write("\n")

        for cons in constants.keys():
            f.write(f"{cons},")
            f.write("\n")
        f.write("]")
        f.write("\n")
        f.write("\n")

        # Declare each matrix element

        for name, mat in zip(list_function_names, sympy_expr):
            write_expr_in_txt_file(name, mat)

        f.close()
