import abc
import numpy as np
import scipy.integrate as sp
from typing import NamedTuple, Any, Optional
import matplotlib.pyplot as plt


class ODEResult(NamedTuple):
    """
    Resultatene fra løsningen av en ODE.

    Args:
        time (np.ndarray): Tidspunktene den er løst for
        solution (np.ndarray): Løsningen til problemet ved de gitte tidene.

    Attributes:
        num_states: Antall stadier i systemet
        num_timepoints: Antall tidspunktsrekker for løsningen.
    """

    time: np.ndarray
    solution: np.ndarray

    @property
    def num_states(self) -> int:
        """Antall stadier for ODEen"""
        return len(self.solution)

    @property
    def num_timepoints(self) -> int:
        """Antall tidspunktrekker for løsningen"""
        return len(self.time)


class InvalidInitialConditionError(RuntimeError):
    pass


class ODEmodel(abc.ABC):
    """
    En abstrakt klasse for alle ODES, en slags 'blueprint' for alle klasser som arver fra ODEmodel.
    Klassen innholder noen abstrakte metoder og abstrakte properties.
    Den innholder også en solve metode og plotte metode.

    Attributes:
        num_states: Antall stadier i systemet.

    Methods:
        __call__: Løsningen for systemet.
        solve: Løser ODE-systemet.
        plot_ode_solution: plotter eller lagrer løsningen for systemet.

    """

    @abc.abstractmethod
    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """Regner ut løsningen til ODE-systemet."""
        pass

    @property
    @abc.abstractmethod
    def num_states(self) -> int:
        """Hvor mange stadier ODE-systemet har."""
        pass

    def _create_result(self, solution: Any) -> ODEResult:
        """
        Lager et Resultat som et objekt fra 'ODEResult'. Ved bruk av solve metoden i ODEmodel.

        Args:
            solution: bør være et objekt som inneholder en rekke med t-er og y-er.
        """
        return ODEResult(time=solution.t, solution=solution.y)

    def solve(
        self, u0: np.ndarray, T: float, dt: float, method: str = "RK45"
    ) -> ODEResult:
        """
        Løser ODE-systemet

        Args:
            u0(np.ndarray): en numpy array som inneholder opprinnelig tilstand.
            T(float): Ende punktet mhp tid.
            dt(float): hvor store hopp du vil ha mellom hvert element i tidsrekken.
            methode (str): hvilken metode du vil at problemet skal bli løst etter.

        Returns:
            Et objekt fra ODEResult som innholder løsningen og tidspunkter.

        Example:
        ExponentialDecay, er en klasse som arver fra ODEmodel, og implementerer en ODE.
        >>> model = ExponentialDecay(0.4)
        >>> result = model.solve(u0=np.array([4.0]), T=10.0, dt=0.01)
        """

        if len(u0) != self.num_states:
            raise InvalidInitialConditionError(
                "Du har gitt for mange eller for få initielle verdier"
                + str(u0)
                + "burde være: "
                + str(self.num_states)
            )

        time_points = np.arange(0, T + dt, dt)
        result = sp.solve_ivp(
            fun=self.__call__, t_span=[0, T], y0=u0, t_eval=time_points, method=method
        )
        return self._create_result(result)


def plot_ode_solution(
    results: ODEResult,
    state_labels: Optional[list[str]] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Funkjsoner plotter løsningen til solve metoden i ODEmodell eller
    lagrer løsningen som en png hvis ønskelig.

    Args:
        resultater(ODEResult): Et objekt man får fra solve metoden.
        state_label(Optional[list[str]]): En liste med strings som blir merknaden i plottet.(frivillig)
        filname(Optional[str]): En string som blir filnavnet til filen.(frivillig)

    Returns:
        returnerer None, men man får plottet enten lagret som fil ellers blir det vist.
    Eksempel:
        ExponentialDecay, er en klasse som arver fra ODEmodel, og implementerer en ODE.
        >>> model = ExponentialDecay(0.4)
        >>> result = model.solve(u0=np.array([4.0]), T=10.0, dt=0.01)
        >>> plot_ode_solution(
        >>> results=result, state_labels=["u"], filename="exponential_decay.png")

    """
    if state_labels is None:
        state_labels = [f"State {i+1}" for i in range(results.num_timepoints)]

    for index, value in enumerate(results.solution):
        plt.plot(results.time, value, label=state_labels[index])

    plt.grid()
    plt.ylabel("Løsning ODE")
    plt.xlabel("Tid")
    plt.legend()
    if filename != None:
        plt.savefig(filename)
    else:
        plt.show()
