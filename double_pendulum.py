import numpy as np
from ode import ODEmodel,plot_ode_solution
from dataclasses import dataclass
from typing import Any, Optional
import matplotlib.pyplot as plt


@dataclass
class DoublePendulumResults:
    """
    Resultater fra løsningen av pendel problemet

    Args:
        time (np.ndarray): Tidsstegene til løsningen.
        solution (np.ndarray): Verdiene til løsningen for de gitte tidspunktene.
        L1 (float): Lengden av av den første snoren til 'DoublePendulum'.
        L2 (float): Lengden av av den andre snoren til 'DoublePendulum'.
        g (float): Tyngdekraften.


    """

    time: np.ndarray
    solution: np.ndarray
    L1: float
    L2: float
    g: float

    @property
    def theta1(self) -> np.ndarray:
        """Løsningen for dtheta1_dt"""
        return self.solution[0]

    @property
    def theta2(self) -> np.ndarray:
        """Løsningen for dtheta2_dt"""
        return self.solution[2]

    @property
    def omega1(self) -> np.ndarray:
        """Løsningen for domega1_dt"""
        return self.solution[1]

    @property
    def omega2(self) -> np.ndarray:
        """Løsningen for domega2_dt"""
        return self.solution[3]

    @property
    def x1(self) -> np.ndarray:
        """x kordinatet til det første pendelet"""
        return self.L1 * np.sin(self.theta1)

    @property
    def y1(self) -> np.ndarray:
        """y kordinatet til det første pendelet"""
        return (-1) * self.L1 * np.cos(self.theta1)

    @property
    def x2(self) -> np.ndarray:
        """x kordinatet til det andre pendelet"""
        return self.x1 + self.L2 * np.sin(self.theta2)

    @property
    def y2(self) -> np.ndarray:
        """y kordinatet til det andre pendelet"""
        return self.y1 - self.L2 * np.cos(self.theta2)

    @property
    def potensiell_energi(self) -> np.ndarray:
        """Potensielle energien til DoublePendulum"""
        return self.g * (self.y1 + self.L1) + self.g * (self.y2 + self.L1 + self.L2)

    @property
    def vx1(self) -> np.ndarray:
        """Hastigheten til det første pendelet i x-planet"""
        return np.gradient(self.x1, self.time)

    @property
    def vy1(self) -> np.ndarray:
        """Hastigheten til det første pendelet i y-planet"""
        return np.gradient(self.y1, self.time)

    @property
    def vx2(self) -> np.ndarray:
        """Hastigheten til det andre pendelet i x-planet"""
        return np.gradient(self.x2, self.time)

    @property
    def vy2(self) -> np.ndarray:
        """Hastigheten til det andre pendelet i y-planet"""
        return np.gradient(self.y2, self.time)

    @property
    def kinetisk_energi(self) -> np.ndarray:
        """kinetiske energi til Doublependulum"""
        return (1 / 2 * (self.vx1**2 + self.vy1**2)) + (
            1 / 2 * (self.vx2**2 + self.vy2**2)
        )

    @property
    def total_energi(self) -> np.ndarray:
        """Den totale energien"""
        return self.potensiell_energi + self.kinetisk_energi


class DoublePendulum(ODEmodel):

    def __init__(self, L1: float = 1, L2: float = 1, g: float = 9.81):
        """
        Initialiserer DoubblePendulum.
        Args:
            L1(float): Lengden på den første snoren som holder en kule.
            L2(float): Lengden på den andre snoren som holder en kule.
            g(float): Tyngdekraften
        """
        self.L1 = L1
        self.L2 = L2
        self.g = g

    @property
    def num_states(self) -> int:
        """Stadier i systemet"""
        return 4

    def _create_result(self, solution: Any) -> DoublePendulumResults:
        """
        Lager et Resultat som et objekt fra 'DoublePendulumResult'. Ved bruk av solve metoden i ODEmodel

        Args:
            solution: bør være et objekt som inneholder en rekke med t-er og y-er.
        """
        return DoublePendulumResults(
            L1=self.L1, L2=self.L2, g=self.g, time=solution.t, solution=solution.y
        )

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """
        Regn ut den deriverte av thetaene og omegaene i tidsteg t.

        Args:
            t(float): hvilket tidspunkt du vil regne ut.
            u(np.ndarray): De verdiene du vil finne den deriverte av.

        Returns:
            Returnerer en np.ndarray med den dervierte av theta1, theta2, omega1 og omega2 i t.
        """

        theta1, omega1, theta2, omega2 = u
        delta_theta = theta2 - theta1

        dtheta1_dt = omega1
        dtheta2_dt = omega2

        dw1_dt_nevner = 2 * self.L1 - self.L1 * np.cos(delta_theta) ** 2
        dw1_dt_teller = (
            self.L1 * omega1**2 * np.sin(delta_theta) * np.cos(delta_theta)
            + self.g * np.sin(theta2) * np.cos(delta_theta)
            + self.L2 * omega2**2 * np.sin(delta_theta)
            - 2 * self.g * np.sin(theta1)
        )

        domega1_dt = dw1_dt_teller / dw1_dt_nevner

        dw2_dt_nevner = 2 * self.L2 - self.L2 * np.cos(delta_theta) ** 2
        dw2_dt_teller = (
            -self.L2 * omega2**2 * np.sin(delta_theta) * np.cos(delta_theta)
            + 2 * self.g * np.sin(theta1) * np.cos(delta_theta)
            - 2 * self.L1 * omega1**2 * np.sin(delta_theta)
            - 2 * self.g * np.sin(theta2)
        )

        domega2_dt = dw2_dt_teller / dw2_dt_nevner

        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])


def plot_energi(
    resultater: DoublePendulumResults, filnavn: Optional[str] = None
) -> None:
    """
    Grafisk fremstilling av pendlenes potensielle, kinetiske og totale energi nivår for gitte tider.
    Hvis filnavn er inkludert i funkjsonkallet blir den lagret som en png-fil.

    Args:
        resultater(PendulumResults): Et objekt man får fra solve metoden.
        filname(Optional[str]): En string som blir filnavnet til filen.(frivillig)

    Returns:
        returnerer None, men man får plottet enten lagret som fil ellers blir det vist.

    """

    plt.plot(resultater.time, resultater.potensiell_energi, label="Potensiell energi")
    plt.plot(resultater.time, resultater.kinetisk_energi, label="Kinetisk energi")
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


def exercise_3d() -> None:
    """ "Løser DoublePendulum og lagrer en grafikk av energien til Pendelet over tid"""
    pendler = DoublePendulum()
    løsning = pendler.solve(
        u0=np.array([np.pi / 6, 0.35, 0, 0]), T=10, dt=0.01, method="Radau"
    )
    plot_energi(resultater=løsning, filnavn="energy_double.png")


if __name__ == "__main__":
    #exercise_3d()

    # model = DoublePendulum()
    # svar = model.solve(u0=np.array([np.pi, 0.35, 0, 0]), T=40.0, dt=0.01)
    # plot_ode_solution(results=svar, state_labels=["u", "a", "b", "c"])

    import matplotlib.animation as animation
    from matplotlib.lines import Line2D
    from matplotlib.text import Text


    def animate_pendulum(results: DoublePendulumResults) -> None:
        """Create an animation of the swinging double pendulum problem.
        
        Args:
            results (DoublePendulumResults): Computed results of the problem
        """
        fig = plt.figure()
        ax = fig.add_subplot(
            111, aspect="equal", autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2)
        )
        ax.grid()

        (line,) = ax.plot([], [], "o-", lw=2)
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        kinetic_energy_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)
        potential_energy_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

        def init() -> tuple[Line2D, Text, Text, Text]:
            """Initialize the animation"""
            line.set_data([], [])
            time_text.set_text("")
            kinetic_energy_text.set_text("")
            potential_energy_text.set_text("")
            return line, time_text, kinetic_energy_text, potential_energy_text

        def animate(i: int) -> tuple[Line2D, Text, Text, Text]:
            """Perform an animation step"""
            line.set_data(
                (0, results.x1[i], results.x2[i]), (0, results.y1[i], results.y2[i])
            )
            time_text.set_text(f"time = {results.time[i]:.1f}")
            kinetic_energy_text.set_text(
                f"kinetic energy = {results.kinetisk_energi[i]:.3f} J"
            )
            potential_energy_text.set_text(
                f"potential energy = {results.potensiell_energi[i]:.3f} J"
            )
            return line, time_text, kinetic_energy_text, potential_energy_text

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(results.time),
            interval=10,
            blit=True,
            init_func=init,
        )
        plt.show()


    def exercise_4() -> None:
        model = DoublePendulum()
        results = model.solve(u0=np.array([np.pi, 0.35, 0, 0]), T=40.0, dt=0.01)
        animate_pendulum(results)
    
    exercise_4()