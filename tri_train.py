"""SAMI trimodal training entrypoint."""

from engines.cli import dispatch


def main(argv=None):
    dispatch("train", argv, trimodal=True)


if __name__ == "__main__":
    main()
