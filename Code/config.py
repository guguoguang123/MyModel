"""Project-level entry point for experiment settings."""

from triwave_gnn.utils.config import DEFAULT_DATA_DIR, DEFAULT_EXP_DIR, training_parser


def get_parse_args(argv=None):
    return training_parser().parse_args(argv)


if __name__ == "__main__":
    print(get_parse_args())
