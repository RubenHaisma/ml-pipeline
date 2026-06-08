"""mlp — the binary. One Typer app, one subcommand per capability.

House rules (enforced in CI):
- every command takes ``--json``
- exit codes are load-bearing (0 ok, non-zero failure)
- no command writes state outside MLflow + ./artifacts
"""

from __future__ import annotations

import typer

from ml_pipeline import __version__
from ml_pipeline.commands.calibrate import calibrate
from ml_pipeline.commands.doctor import doctor
from ml_pipeline.commands.drift import drift
from ml_pipeline.commands.eval import eval
from ml_pipeline.commands.infer import infer
from ml_pipeline.commands.train import train

app = typer.Typer(
    name="mlp",
    help="Classic ML done right — train, calibrate, monitor drift, tracked in MLflow.",
    no_args_is_help=True,
    add_completion=False,
)

app.command()(doctor)
app.command()(train)
app.command()(eval)
app.command()(calibrate)
app.command()(drift)
app.command()(infer)


@app.command()
def version(json_out: bool = typer.Option(False, "--json")) -> None:
    """Print the mlp version."""
    if json_out:
        typer.echo(f'{{"version": "{__version__}"}}')
    else:
        typer.echo(__version__)


if __name__ == "__main__":
    app()
