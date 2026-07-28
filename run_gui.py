"""Entry point: `python run_gui.py` launches the 1D Fokker-Planck solver GUI.

Coefficient .npz files (see coefficients/README.md) are looked up in the
'coefficients' folder next to this script.
"""
from fp1d.gui import main

if __name__ == '__main__':
    main()
