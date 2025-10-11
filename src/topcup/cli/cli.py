import click
from topcup.cli.log import setup_logging
from topcup.cli.main import train, inference, calculate_score


click.MAX_WIDTH = 120

# Make --version work without extra boilerplate
from importlib.metadata import version, PackageNotFoundError

def _pkg_version():
    try:
        return version("topcup")
    except PackageNotFoundError:
        return "0.0.0-dev"

class State:  # shared state via Click context
    def __init__(self, verbose):
        self.verbose = verbose


@click.group(context_settings={"help_option_names": ["-h", "--help"]},
             #invoke_without_command=True, no_args_is_help=True, sort_commands=False
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv).")
@click.version_option(_pkg_version())
@click.pass_context
def cli(ctx, verbose):
    """
    topcup: a top crypet u-net picker 
    """
    setup_logging(verbose)
    ctx.obj = State(verbose=verbose)



# Register subcommands
cli.add_command(train)
cli.add_command(inference)
cli.add_command(calculate_score)



if __name__ == "__main__":
    cli(prog_name="topcup")