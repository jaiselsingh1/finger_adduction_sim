import numpy as np
import sympy as sp


# ==========================================================
# 1. SYMBOLIC DEFINITIONS
# ==========================================================

# Joint variables
theta1, theta2, theta3 = sp.symbols('theta1 theta2 theta3', real=True)

# Lengths
L1, L2, L3 = sp.symbols('L1 L2 L3', positive=True)

# Anatomical coupling
theta4 = theta3/2

# DH transformation
def DH(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),               d],
        [0,              0,                           0,                            1]
    ])

A1 = DH(0, sp.pi/2, 0, theta1)
A2 = DH(L1, 0,      0, theta2)
A3 = DH(L2, 0,      0, theta3)
A4 = DH(L3, 0,      0, theta4)

T = sp.simplify(A1 * A2 * A3 * A4)

# Local planar coordinates (before MCP abduction rotation)
px_p = sp.simplify(T[0,3]*sp.cos(theta1) + T[1,3]*sp.sin(theta1))
pz_p = sp.simplify(T[2,3])


# ==========================================================
# 2. NUMERICAL FORWARD KINEMATICS
# ==========================================================

def finger_forward_kinematics(theta1v, theta2v, theta3v, theta4v, L1v, L2v, L3v):

    psi = theta2v + theta3v
    phi = theta2v + 1.5 * theta3v

    px_p = L1v*np.cos(theta2v) + L2v*np.cos(psi) + L3v*np.cos(phi)
    pz   = L1v*np.sin(theta2v) + L2v*np.sin(psi) + L3v*np.sin(phi)

    px = px_p * np.cos(theta1v)
    py = px_p * np.sin(theta1v)

    return px, py, pz


# ==========================================================
# 3. NUMERICAL INVERSE KINEMATICS USING nsolve
# ==========================================================

# Create lambdified symbolic functions
px_fun = sp.lambdify((theta1, theta2, theta3, L1, L2, L3), px_p, 'numpy')
pz_fun = sp.lambdify((theta1, theta2, theta3, L1, L2, L3), pz_p, 'numpy')

def finger_inverse_kinematics(px, py, pz, L1v, L2v, L3v,
                              theta2_init=0.5, theta3_init=0.5):

    # MCP abduction
    theta1_val = float(np.arctan2(py, px))

    # planar coordinates
    rho_val = float(np.sqrt(px**2 + py**2))
    z_val   = float(pz)

    # Define the system for nsolve
    eq1 = sp.Eq(px_p.subs({L1:L1v, L2:L2v, L3:L3v, theta1:theta1_val}), rho_val)
    eq2 = sp.Eq(pz_p.subs({L1:L1v, L2:L2v, L3:L3v, theta1:theta1_val}), z_val)

    sol = sp.nsolve(
        (eq1, eq2),
        (theta2, theta3),
        (theta2_init, theta3_init),
        tol=1e-12,
        maxsteps=50
    )

    theta2_val = float(sol[0])
    theta3_val = float(sol[1])
    theta4_val = 0.5 * theta3_val

    return theta1_val, theta2_val, theta3_val, theta4_val


# ==========================================================
# 4. TEST FK → IK → FK
# ==========================================================

if __name__ == "__main__":

    L1v, L2v, L3v = np.sqrt(2), 2.0, np.sqrt(2)

    # Ground truth angles
    theta1_true = np.deg2rad(20)
    theta2_true = np.deg2rad(10)
    theta3_true = np.deg2rad(50)
    theta4_true = 0.5 * theta3_true

    # FK -> position
    px, py, pz = finger_forward_kinematics(theta1_true, theta2_true,
                                           theta3_true, theta4_true,
                                           L1v, L2v, L3v)
    print("FK position:", px, py, pz)

    # IK -> recovered angles
    th1, th2, th3, th4 = finger_inverse_kinematics(px, py, pz, L1v, L2v, L3v,
                                                   theta2_init=0.2,
                                                   theta3_init=0.5)

    print("IK angles (deg):", np.degrees([th1, th2, th3, th4]))

    # FK again
    px2, py2, pz2 = finger_forward_kinematics(th1, th2, th3, th4, L1v, L2v, L3v)
    print("FK(IK):", px2, py2, pz2)

    print("Error:", px2-px, py2-py, pz2-pz)