import functools
import multiprocessing
import pathlib
import subprocess

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


model_eval_cmd_tmpl = f"poetry run python main.py eval -m '{{model_file}}' {m.to_base_units().magnitude} {k.to_base_units().magnitude} {g.to_base_units().magnitude} {{t}} {vt.to_base_units().magnitude}"


def run_model(model_file):
    results_file = model_file.with_suffix(".txt")
    cmd_tmpl = functools.partial(model_eval_cmd_tmpl.format, model_file=model_file)
    dt = 0.1
    N = 100
    current_process = multiprocessing.current_process()
    pbar_pos = current_process._identity[0] - 1
    with open(results_file, "w") as f:
        idxs = range(N)
        with tqdm(total=N, desc=str(model_file), position=pbar_pos) as pbar:
            for i in range(N):
                t = dt * i
                y = fall_distance(t)
                cmd = cmd_tmpl(t=t)
                model_y = (
                    subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
                )
                f.write(f"{t} {y} {model_y}\n")
                pbar.update(1)


# this has to be limited, otherwise cuda runs out of memory to allocate
with multiprocessing.Pool(
    2, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)
) as p:
    p.map(run_model, pathlib.Path().glob("out/*.keras"))
