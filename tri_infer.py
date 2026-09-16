"""SAMI trimodal inference entrypoint."""

from engines.cli import dispatch


def main(argv=None):
    dispatch("infer", argv, trimodal=True)


if __name__ == "__main__":
    main()
