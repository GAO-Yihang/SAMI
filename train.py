"""SAMI training entrypoint."""

from engines.cli import dispatch


def main(argv=None):
    dispatch("train", argv)


if __name__ == "__main__":
    main()
