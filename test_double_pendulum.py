import pytest
import numpy as np
from double_pendulum import DoublePendulum


def test_derivatives_at_rest_is_zero() -> None:
    """Tester at de deriverte er null"""
    t = 0.0
    tol = 1e-14
    pendler = DoublePendulum()
    løsning = pendler(t, np.array([0, 0, 0, 0]))
    assert abs(løsning[0] - 0) < tol
    assert abs(løsning[1] - 0) < tol
    assert abs(løsning[2] - 0) < tol
    assert abs(løsning[3] - 0) < tol


def test_solve_pendulum_ode_with_zero_ic() -> None:
    """Tester at alle løsningen er null ved null initalbetingelse"""
    tol = 1e-14
    pendler = DoublePendulum()
    løsning = pendler.solve(u0=np.array([0, 0, 0, 0]), T=10, dt=0.01)
    theta1, omega1, theta2, omega2 = løsning.solution
    assert np.all(abs(theta1) < tol)
    assert np.all(abs(theta2) < tol)
    assert np.all(abs(omega1) < tol)
    assert np.all(abs(omega2) < tol)


@pytest.mark.parametrize(
    "theta1, theta2, forventet",
    [
        (0, 0, 0),
        (0, 0.5, 3.386187037),
        (0.5, 0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ],
)
def test_domega1_dt(theta1: float, theta2: float, forventet: float) -> None:
    """Testing at  domega_1/dt er implementert riktig"""
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = model(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, forventet)


@pytest.mark.parametrize(
    "theta1, theta2, forventet",
    [
        (0, 0, 0.0),
        (0, 0.5, -7.704787325),
        (0.5, 0, 6.768494455),
        (0.5, 0.5, 0.0),
    ],
)
def test_domega2_dt(theta1: float, theta2: float, forventet: float) -> None:
    """Testing at domega_2/dt er implementert riktig"""
    model = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = model(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, forventet)


def test_solve_double_pendulum_function_zero_ic():
    """Tester at kordinatene for pendlene er riktig for initalbetingelse null"""
    tol = 1e-14
    pendler = DoublePendulum()
    løsning = pendler.solve(u0=np.array([0, 0, 0, 0]), T=10, dt=0.01)
    assert np.all(abs(løsning.x1) < tol)
    assert np.all(abs(løsning.x2) < tol)
    assert np.all(abs(løsning.y1) - løsning.L1 < tol)
    assert np.all(abs(løsning.y1) - (løsning.L1 + løsning.L2) < tol)
