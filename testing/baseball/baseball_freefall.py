import functools
import multiprocessing
import os
import pathlib
import subprocess

import numpy as np
import torch

torch.cuda.is_available = lambda: False
torch.device("cpu")

os.environ["KERAS_BACKEND"] = "torch"

import click
import keras
import numpy
import pint
from tqdm import tqdm

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


# constants
m = Q_(145, "g")
rho = Q_(1.293, "kg/m^3")
c = Q_(9, "in")
g = Q_(9.8, "m/s^2")
pi = numpy.pi
vt = Q_(95, "mph")

# derived values
A = c**2 / 4 / pi
C = 2 * m * g / vt / rho / A
C = C.to_base_units()

k = 0.5 * C * rho * A
tau = (vt / g).to("s").magnitude


def fall_distance(t):
    return tau * vt.to("m/s").magnitude * numpy.log(numpy.cosh(t / tau))


@click.command()
@click.argument("model_files", type=pathlib.Path, nargs=-1)
def main(model_files):
    for model_file in model_files:
        print(f"Running {model_file}")
        results_file = model_file.with_suffix(".txt")
        model = keras.saving.load_model(model_file)

        dt = 0.1
        N = 100
        ts = [dt * i for i in range(N)]
        inputs = np.array(
            [
                [
                    m.to_base_units().magnitude,
                    k.to_base_units().magnitude,
                    g.to_base_units().magnitude,
                    t,
                    vt.to_base_units().magnitude,
                ]
                for t in ts
            ]
        )

        ys = [fall_distance(t) for t in ts]
        model_ys = model.predict(inputs, verbose=False)
        with open(results_file, "w") as f:
            f.write("#t[s] y[m] model_y[m]")
            for t, y, model_y in zip(ts, ys, model_ys):
                f.write(f"{t} {y} {model_y[0]}\n")


if __name__ == "__main__":
    main()
