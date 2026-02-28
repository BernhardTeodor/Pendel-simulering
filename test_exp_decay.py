from exp_decay import ExponentialDecay
from ode import ODEmodel, InvalidInitialConditionError, ODEResult, plot_ode_solution
import pytest
import numpy as np
from pathlib import Path


def test_negative_decay_raises_ValueError_constructor() -> None:
    """Tester at det blir hevet en ValueError hvis det blir lagt inn et negativt tall"""
    with pytest.raises(ValueError):
        ExponentialDecay(-1.5)


@pytest.mark.parametrize("u, decay, svar", [(np.array([3.2]), 0.4, -1.28)])
def test_rhs(u: np.array, decay: float, svar: float) -> None:
    """
    Tester call metoden i ExponentialDecay.
    Bruker paramterizetest fra pytest.

    """
    tol = 1e-14
    t = 0.0
    test_model = ExponentialDecay(decay)
    du_dt = test_model(t, u)
    assert abs(du_dt[0] - svar) < tol


def test_negative_decay_raises_ValueError() -> None:
    """Tester at man ikke kan endre decay parameteret til et negativ tall"""
    with pytest.raises(ValueError):
        model = ExponentialDecay(0.4)
        model.decay = -1


def test_num_states() -> None:
    """
    Tester at ExponentialDecay har 1 state variabel
    og at det kommer en AttributeError hvis den blir endret.
    """
    model = ExponentialDecay(0.4)

    assert model.num_states == 1

    with pytest.raises(AttributeError):
        model.num_states = 20


def test_create_solution() -> None:
    """Tester at det heves en AttributeError hvis solution ikke har .t og .y"""
    with pytest.raises(AttributeError):
        modell = ExponentialDecay(0.4)
        modell._create_result(["har denne .y og .t"])


def test_solve_with_different_number_of_initial_states() -> None:
    """Tester at solvemetoden i ODEmodel hever en 'InvalidInitialConditionError'"""

    with pytest.raises(InvalidInitialConditionError):
        modell = ExponentialDecay(0.2)
        resultat = modell.solve(u0=np.array([1, 2]), T=10, dt=0.1)


@pytest.mark.parametrize(
    "decay, u0, T, dt",
    (
        [0.4, np.array([3.2]), 10.0, 0.01],
        [0.8, np.array([6.0]), 40.0, 0.4],
        [0.5, np.array([4.2]), 365, 1.0],
    ),
)
def test_solve_time(decay: float, u0: np.ndarray, T: float, dt: float) -> None:
    """Tester at solve funksjonen returnerer
    en time array som har riktige dimensjoner"""
    tol = 1e-14
    modell = ExponentialDecay(decay)
    resulat = modell.solve(u0=u0, T=T, dt=dt)
    assert resulat.time[0] == 0
    assert abs(resulat.time[-1] - T) < tol
    test_av_dt = resulat.time[2] - resulat.time[1]
    assert abs(test_av_dt - dt) < tol


@pytest.mark.parametrize(
    "decay, u0, T, dt",
    (
        [0.4, np.array([3.2]), 100, 0.1],
        [0.1, np.array([2.1]), 2000, 0.5],
        [0.8, np.array([89.2]), 10, 0.05],
    ),
)
def test_solve_solution(decay: float, u0: np.ndarray, T: float, dt: float) -> None:
    """
    Tester at resultatet fra solve metoden i ODEmodel.
    Sjekker at den relative feilen er mindre enn et prosent
    fra den eksakte løsningen.
    """

    def eksakt_losning(u0: np.ndarray, decay: float, t: np.ndarray) -> np.ndarray:
        """
        Kalkulere den eksakte løsningen til Exponential decay.

        Args:
            u0(np.ndarray):initial tilstand
            decay(float): en parameter mellom 0 og 1
            t(np.ndarray): som har alle tidsintervallene, man skal kalkulere for

        Returns:
            returnerer en np.ndarray med alle løsningene for t.

        Eksempel:
            >>> eksakt_løsning(0.2, 2.4, np.array([1,3,4,5,6]))
        """
        losning = u0 * np.exp((-1) * decay * t)
        return losning

    tidspunkter = np.arange(0, T + dt, dt)

    y_eksakt = eksakt_losning(u0, decay, tidspunkter)

    modell = ExponentialDecay(decay)
    resultat = modell.solve(u0, T, dt)

    y_omtrent = resultat.solution

    relative_feil = np.linalg.norm(y_omtrent - y_eksakt) / np.linalg.norm(y_eksakt)

    assert relative_feil < 0.01


def test_ODEResult() -> None:
    """Tester at ODEResult returner riktig antall tilstander og antall tidspunkter"""

    results = ODEResult(time=np.array([0, 1, 2]), solution=np.zeros((2, 3)))

    assert results.num_states == 2
    assert results.num_timepoints == 3


def test_plot_ode_solution_saves_file() -> None:
    """
    Tester at 'plot_ode_solution', lager en fil.
    Når denne testen kjøres blir filen slettet etterpå
    """

    filename = Path("test_plot_ode_midlertidig_skal_slettes.png")
    if filename.is_file():
        filename.unlink()

    def function_that_creates_a_file(filename: Path) -> None:
        """En funkjson som kaller på 'plot_ode_solution'."""
        modell = ExponentialDecay(0.4)
        resultat = modell.solve(u0=np.array([4.0]), T=10.0, dt=0.01)
        plot_ode_solution(resultat, filename=filename)
        return

    function_that_creates_a_file(filename)

    assert filename.is_file()
    filename.unlink()
