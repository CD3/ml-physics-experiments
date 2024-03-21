import click
import pandas as pd
import os
import numpy as np

import cream.simple_1d as s1d

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import losses, optimizers, callbacks, layers


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "-o",
    "--output",
    type=click.File("wb"),
    default="out/data.csv",
    help="File to write generated data to",
)
@click.argument("rows", type=int)
def gendat(output, rows):
    """Generate dummy data for use in training the model or validating it"""
    s1d.generate(rows).to_csv(output)


@click.command()
@click.option(
    "-t",
    "--training-data",
    type=click.File("rb"),
    default="out/training.csv",
    help="File to read training data from",
)
@click.option(
    "-s",
    "--testing-data",
    type=click.File("rb"),
    default="out/testing.csv",
    help="File to read testing data from",
)
@click.option(
    "-m",
    "--model-file",
    type=click.Path(),
    default="out/final.keras",
    help="File to write the final model to",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=20,
    help="The number of epochs to train for",
)
def train(training_data, testing_data, model_file, epochs):
    """Train a model and write its weights out after each epoch to a file"""
    training = pd.read_csv(training_data)
    testing = pd.read_csv(testing_data)

    training["k"], testing["k"] = s1d.K_(
        training["drag_coefficient"],
        training["fluid_density"],
        training["cross_sectional_area"],
    ), s1d.K_(
        testing["drag_coefficient"],
        testing["fluid_density"],
        testing["cross_sectional_area"],
    )

    training["terminal_velocity"], testing["terminal_velocity"] = s1d.Vt_(
        training["mass"],
        training["gravitational_acceleration"],
        training["drag_coefficient"],
        training["fluid_density"],
        training["cross_sectional_area"],
    ), s1d.Vt_(
        testing["mass"],
        testing["gravitational_acceleration"],
        testing["drag_coefficient"],
        testing["fluid_density"],
        testing["cross_sectional_area"],
    )

    training_x, training_y = (
        training[
            ["mass", "k", "gravitational_acceleration", "time", "terminal_velocity"]
        ],
        training["relative_position"],
    )

    testing_x, testing_y = (
        testing[
            ["mass", "k", "gravitational_acceleration", "time", "terminal_velocity"]
        ],
        testing["relative_position"],
    )

    model = keras.Sequential(
        [
            layers.Input(shape=(5,)),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1),
        ]
    )

    model.compile(
        loss=losses.Huber(delta=10.0),
        optimizer=optimizers.Adam(learning_rate=1e-3),
    )

    model.fit(
        training_x,
        training_y,
        batch_size=2048,
        epochs=epochs,
        validation_split=0.10,
        callbacks=[
            callbacks.ModelCheckpoint(filepath="out/{epoch}.keras"),
        ],
    )

    model.evaluate(testing_x, testing_y)
    model.save(model_file)


@click.command()
@click.argument("m", type=float)
@click.argument("k", type=float)
@click.argument("g", type=float)
@click.argument("t", type=float)
@click.argument("vt", type=float)
@click.option(
    "-m",
    "--model-file",
    type=click.Path(),
    default="out/final.keras",
    help="File to read the model from",
)
def eval(m, k, g, t, vt, model_file):
    """Evaluate a given model against the parameters"""
    model = keras.saving.load_model(model_file)
    click.echo(model.predict(np.array([[m, k, g, t, vt]]), verbose=0)[0][0])


@click.command()
@click.option(
    "-s",
    "--testing-data",
    type=click.File("rb"),
    default="out/testing.csv",
    help="File to read testing data from",
)
@click.option(
    "-m",
    "--model-file",
    type=click.Path(),
    default="out/final.keras",
    help="File to write the final model to",
)
def test(testing_data, model_file):
    """Evaluate a given model against a set of testing data"""
    testing = pd.read_csv(testing_data)

    testing["k"] = s1d.K_(
        testing["drag_coefficient"],
        testing["fluid_density"],
        testing["cross_sectional_area"],
    )

    testing["terminal_velocity"] = s1d.Vt_(
        testing["mass"],
        testing["gravitational_acceleration"],
        testing["drag_coefficient"],
        testing["fluid_density"],
        testing["cross_sectional_area"],
    )

    testing_x, testing_y = (
        testing[
            ["mass", "k", "gravitational_acceleration", "time", "terminal_velocity"]
        ],
        testing["relative_position"],
    )

    model = keras.saving.load_model(model_file)
    model.evaluate(testing_x, testing_y)


cli.add_command(gendat)
cli.add_command(train)
cli.add_command(eval)
cli.add_command(test)

if __name__ == "__main__":
    cli()
