import pytest
import numpy as np
from pendulum import Pendulum


@pytest.mark.parametrize(
    "L, omega, theta, svar",
    [(1.42, 0.35, np.pi / 6, np.array([0.35, -3.45422535211267]))],
)
def test_rhs_pendulum(L: float, omega: float, theta: float, svar: np.ndarray) -> None:
    """Tester call metoden i Pendulum"""

    t = 0.0
    tol = 1e-14
    pendel = Pendulum(L)
    løsning = pendel(t, np.array([theta, omega]))
    assert abs(løsning[0] - svar[0]) < tol
    assert abs(løsning[1] - svar[1]) < tol


def test_pendel_equlibrium() -> None:
    """Tester at pendelen hviler når thete og omega er null"""

    t = 0.0
    tol = 1e-14
    theta = 0.0
    omega = 0.0
    pendel = Pendulum()
    løsning = pendel(t, np.array([theta, omega]))
    assert abs(løsning[0] - 0) < tol
    assert abs(løsning[1] - 0) < tol


def test_solve_pendulum_ode_with_zero_ic() -> None:
    """
    Tester at løsningen er liste med nuller
    hvis initalbetingelsen er np.array([0,0])
    """
    tol = 1e-14
    pendel = Pendulum()
    løsning = pendel.solve(u0=np.array([0, 0]), T=10, dt=0.01)
    theta, omega = løsning.solution
    assert np.all(abs(theta) < tol)
    assert np.all(abs(omega) < tol)


def test_solve_pendulum_function_zero_ic() -> None:
    """Tester at x og y kordinatene er null"""
    tol = 1e-14
    pendel = Pendulum()
    løsning = pendel.solve(u0=np.array([0, 0]), T=10, dt=0.01)
    assert np.all(abs(løsning.x) < tol)
    assert np.all(abs(løsning.y) - løsning.L < tol)
