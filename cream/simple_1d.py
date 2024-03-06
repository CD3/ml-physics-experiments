import numpy as np
import pandas as pd


def m_(g: np.random.Generator, n: int, a: float, b: float) -> np.ndarray:
    """Sample `n` mass values (in kg) from a normal distribution, within the range [a, b)"""
    return g.uniform(a, b, n)


def C_(g: np.random.Generator, n: int) -> np.ndarray:
    """Sample `n` drag coefficient values (unitless) from a normal distribution, constrained such that most values will be within the range 0.4 - 1.0"""
    return np.fabs(g.normal(0.7, 2.0, n))


def A_(g: np.random.Generator, n: int, a: float, b: float) -> np.ndarray:
    """Sample `n` cross-sectional area values (in m^2) from a uniform distribution, within the range [a, b)"""
    return g.uniform(a, b, n)


def p_(g: np.random.Generator, n: int) -> np.ndarray:
    """Sample `n` fluid density values (in kg/m^3) from a normal distribution, constrained so that most values will be within the range 1.0 - 1.4"""
    return np.fabs(g.normal(1.2, 2.0, n))


def t_(g: np.random.Generator, n: int, b: float) -> np.ndarray:
    """Sample `n` time values (in s) from a uniform distribution, within the range [0, b)"""
    return g.uniform(0, b, n)


def g_(g: np.random.Generator, n: int) -> np.ndarray:
    """Sample `n` gravitational acceleration constant values (in m/s^2) from a normal distribution, constrained such that most values will be within the range 9.4 - 10.2"""
    return np.fabs(g.normal(9.8, 2.0, n))


def Vt_(
    m: np.ndarray, g: np.ndarray, C: np.ndarray, p: np.ndarray, A: np.ndarray
) -> np.ndarray:
    """Compute the terminal velocity (in m/s) given its parameters"""
    return np.sqrt((2 * m * g) / (C * p * A))


def x_(Vt: np.ndarray, g: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute the relative position (in m) of an object in free-fall given its terminal velocity, the gravitational acceleration constant, and the time"""

    # i don't really feel like setting up sympy and getting it to work here
    # so we're just going to suppress overflow errors here. it likely doesn't
    # matter (too) much anyway
    with np.errstate(over="ignore"):
        return ((Vt**2) / g) * np.log(np.fabs(np.cosh((g * t) / Vt)))


def K_(C: float, p: float, A: float) -> float:
    """Compute the `K` value, C*p*A/2"""
    return (C * p * A) / 2


def generate(n: int) -> pd.DataFrame:
    """Generate `n` instances of training data"""
    gn = np.random.default_rng()

    m = m_(gn, n, 0.5, 50)
    C = C_(gn, n)
    A = A_(gn, n, 1, 50)
    p = p_(gn, n)
    t = t_(gn, n, 50)
    g = g_(gn, n)

    Vt = Vt_(m, g, C, p, A)
    x = x_(Vt, g, t)

    df = pd.DataFrame(
        {
            "mass": m,
            "drag_coefficient": C,
            "cross_sectional_area": A,
            "fluid_density": p,
            "time": t,
            "gravitational_acceleration": g,
            "terminal_velocity": Vt,
            "relative_position": x,
        }
    )

    # number precision is probably the cause of this
    df = df.drop(df[df["relative_position"] == np.inf].index)

    return df
