"""
Entry point:  python -m fingerprint_matcher_v2

Launches the PyQt5 GUI for latent fingerprint matching.
"""

import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from fingerprint_matcher_v2.gui.main_window import MainWindow


def main():
    # Set up root logging
    log_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (INFO)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_fmt)
    root.addHandler(ch)

    app = QApplication(sys.argv)
    app.setApplicationName("Fingerprint Matcher v2")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
