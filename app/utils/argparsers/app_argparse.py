# app/utils/argparsers/app_argparse.py

import argparse

from app.utils.helpers import boolean


class AppArgparse:

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="")  # TODO
        # TODO
        parser.add_argument(
            "--argument",
            type=str,
            default="",
            help="",
        )
        args, _ = parser.parse_known_args()
        return args
