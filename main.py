import click
import torch
from torch import nn
from torch.utils.data import DataLoader

import cream.simple_1d as s1d
import cream.network as network
from cream.network import CSVDataset, NeuralNetwork


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "-o",
    "--output",
    type=click.File("wb"),
    default="data.csv",
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
    default="data.csv",
    help="File to read training data from",
)
@click.option(
    "-s",
    "--test-data",
    type=click.File("rb"),
    default="data.csv",
    help="File to read test data from",
)
@click.option(
    "-m",
    "--model-file",
    type=click.Path(),
    default="weights.pth",
    help="File to write weights to",
)
@click.option(
    "-l",
    "--learning-rate",
    type=float,
    default=1e-4,
    help="The learning rate of the model",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=5,
    help="The number of epochs to train for",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=64,
    help="The batch size to train with",
)
def train(training_data, test_data, model_file, learning_rate, epochs, batch_size):
    """Train a model and write its weights out to a file"""

    train_loader = DataLoader(
        CSVDataset(training_data), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(CSVDataset(test_data), batch_size=batch_size, shuffle=True)

    model = NeuralNetwork()

    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        click.echo(f"epoch {t+1}:")
        network.train(train_loader, model, loss_fn, optimizer, batch_size)
        network.test(test_loader, model, loss_fn)

    click.echo(f"finished {epochs} epochs")
    torch.save(model.state_dict(), model_file)


@click.command()
@click.argument("m", type=float)
@click.argument("k", type=float)
@click.argument("g", type=float)
@click.argument("t", type=float)
@click.option(
    "-m",
    "--model-file",
    type=click.Path(),
    default="weights.pth",
    help="File to read weights from",
)
def eval(m, k, g, t, model_file):
    """Evaluate a given model against the parameters"""

    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    click.echo(model(torch.FloatTensor([[m, k, g, t]]))[0][0].item())


cli.add_command(gendat)
cli.add_command(train)
cli.add_command(eval)

if __name__ == "__main__":
    cli()
