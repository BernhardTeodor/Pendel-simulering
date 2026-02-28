import numpy as np
from ode import ODEmodel, plot_ode_solution
from dataclasses import dataclass
from typing import Any, Optional
import matplotlib.pyplot as plt


@dataclass
class PendulumResults:
    """
    Resultater fra løsningen av pendel problemet

    Args:
        time (np.ndarray): Tidsstegene til løsningen.
        solution (np.ndarray): Verdiene til løsningen for de gitte tidspunktene.
        L (float): Lengden av snoren til pendelen.
        g (float): Tyngdekraften.

    Attributes:
        theta: Løsningen for theta.
        omega: Løsningen for omega.
        x: X kordinatet.
        y: Y kordinatet.
        potensiell_energi: Pendeles potensiell energi.
        vx: Hastigheten til pendelet i x.
        vy: Hastigheten til pendelet i y.
        kinetisk_ernergi: kinetisk ernergien til pendelet
        total_energi: pendelets totale energi.

    """

    time: np.ndarray
    solution: np.ndarray
    L: float
    g: float

    @property
    def theta(self) -> np.ndarray:
        """Løsningen for theta."""
        return self.solution[0]

    @property
    def omega(self) -> np.ndarray:
        """Løsningen for omega."""
        return self.solution[1]

    @property
    def x(self) -> np.ndarray:
        """x kordinatet til pendelet."""
        return self.L * np.sin(self.theta)

    @property
    def y(self) -> np.ndarray:
        """y kordinatet til pendelet."""
        return (-1) * self.L * np.cos(self.theta)

    @property
    def potensiell_energi(self) -> np.ndarray:
        """Potensielle energien til pendelet."""
        return self.g * (self.y + self.L)

    @property
    def vx(self) -> np.ndarray:
        """Hastigheten til pendelet i x-planet."""
        return np.gradient(self.x, self.time)

    @property
    def vy(self) -> np.ndarray:
        """Hastigheten til pendelet i y-planet."""
        return np.gradient(self.y, self.time)

    @property
    def kinetisk_ernergi(self) -> np.ndarray:
        """Pendelets kinetiske energi."""
        return 1 / 2 * (self.vx**2 + self.vy**2)

    @property
    def total_energi(self) -> np.ndarray:
        """Pendelets totale energi."""
        return self.potensiell_energi + self.kinetisk_ernergi


class Pendulum(ODEmodel):
    """
    Klassen representerer et Pendel ODE-system.

    Den inneholder informasjon om systemet og metode for å lese det.

     Args:
        L(float): Lengden på snoren som holder en kule.
        g(float): Tyngdekraften

    Attributes:
        num_states(int): Antall stadier i ODE-systemet.

    Methods:
        __call__: Regner ut løsningen til Pendulum-systemet.
        solve: Løser ODE-systemet.
        plot_ode_solution: Viser eller lagrer plottet til løsningen av systemet.

    """

    def __init__(self, L: float = 1, g: float = 9.81):
        """
        Initialiserer Pendulum.
        Args:
            L(float): Lengden på snoren som holder en kule.
            g(float): Tyngdekraften
        """

        self.L = L
        self.g = g

    @property
    def num_states(self) -> int:
        """Antall stadier i systemet."""
        return 2

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Regn ut den deriverte av theta og omega i tidsteg t.

        Args:
            t(float): hvilket tidspunkt du vil regne ut.
            u(np.ndarray): De verdiene du vil finne den deriverte av.

        Returns:
            Returnerer en np.ndarray med den dervierte av theta og omega i t.
        """
        theta, omega = u
        return np.array([omega, -(self.g / self.L) * np.sin(theta)])

    def _create_result(self, solution: Any) -> PendulumResults:
        """
        Lager et Resultat som et objekt fra 'PendulumResult'. Ved bruk av solve metoden i ODEmodel

        Args:
            solution: bør være et objekt som inneholder en rekke med t-er og y-er.
        """
        return PendulumResults(L=self.L, g=self.g, time=solution.t, solution=solution.y)


class DampedPendulum(Pendulum):
    """
    Klassen representerer et 'damped'-Pendel ODE-system.

    Den inneholder informasjon om systemet og metode for å lese det.

     Args:
        B(float): dempingen av pendelet
        L(float): Lengden på snoren som holder en kule.
        g(float): Tyngdekraften

    Attributes:
        num_states(int): Antall stadier i ODE-systemet.

    Methods:
        __call__: Regner ut løsningen til Pendulum-systemet.
        solve: Løser ODE-systemet.
        plot_ode_solution: Viser eller lagrer plottet til løsningen av systemet.

    """

    def __init__(self, B: float, L: float = 1, g: float = 9.81):
        """
        Initialiserer Pendulum.
        Args:
            L(float): Lengden på snoren som holder en kule.
            g(float): Tyngdekraften
        """

        self.B = B
        self.L = L
        self.g = g

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Regn ut den deriverte av theta og omega i tidsteg t.

        Args:
            t(float): hvilket tidspunkt du vil regne ut.
            u(np.ndarray): De verdiene du vil finne den deriverte av.

        Returns:
            Returnerer en np.ndarray med den dervierte av theta og omega i t.
        """

        theta, omega = u
        return np.array([omega, -(self.g / self.L) * np.sin(theta) - self.B * omega])


def plot_energi(resultater: PendulumResults, filnavn: Optional[str] = None) -> None:
    """
    Grafisk fremstilling av pendelens potensielle, kinetiske og totale energi nivår for gitte tider.
    Hvis filnavn er inkludert i funkjsonkallet blir den lagret som en png-fil.

    Args:
        resultater(PendulumResults): Et objekt man får fra solve metoden.
        filname(Optional[str]): En string som blir filnavnet til filen.(frivillig)

    Returns:
        returnerer None, men man får plottet enten lagret som fil ellers blir det vist.

    """

    plt.plot(resultater.time, resultater.potensiell_energi, label="Potensiell energi")
    plt.plot(resultater.time, resultater.kinetisk_ernergi, label="Kinetisk ernergi")
    plt.plot(resultater.time, resultater.total_energi, label="Total energi")

    plt.grid()
    plt.ylabel("Energi")
    plt.xlabel("Tid")
    plt.title("Pendelens energi")
    plt.legend()
    if filnavn != None:
        plt.savefig(filnavn)
    else:
        plt.show()


def exercise_2b() -> None:
    """Løser ODEen for gitte betingelser og lagrer plottet som en png."""

    pendel = Pendulum()
    løsning = pendel.solve(u0=np.array([np.pi / 6, 0.35]), T=10.0, dt=0.01)
    num_labs = [r"$\theta$", r"$\omega$"]
    plot_ode_solution(
        results=løsning, state_labels=num_labs, filename="exercise_2b.png"
    )


def exercise_2g() -> None:
    """Løser Pendulum og lagrer en grafikk av energien til Pendelet over tid."""
    pendel = Pendulum()
    løsning = pendel.solve(u0=np.array([np.pi / 6, 0.35]), T=10.0, dt=0.01)
    plot_energi(resultater=løsning, filnavn="energy_single.png")


def exercise_2h() -> None:
    """Løser DampedPendulum og lagrer en grafikk av energien til Pendelet over tid."""
    pendel = DampedPendulum(B=1)
    løsning = pendel.solve(u0=np.array([np.pi / 6, 0.35]), T=10.0, dt=0.01)
    plot_energi(resultater=løsning, filnavn="energy_damped.png")


if __name__ == "__main__":
    exercise_2b()
    exercise_2g()
    exercise_2h()
