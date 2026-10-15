"""Allow ``python -m video_intel`` as an alternative to the console script."""

from video_intel.cli.main import cli

if __name__ == "__main__":
    cli()
