"""SAMI inference entrypoint."""

from engines.cli import dispatch


def main(argv=None):
    dispatch("infer", argv)


if __name__ == "__main__":
    main()
