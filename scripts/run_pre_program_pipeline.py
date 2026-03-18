"""Run the pre-program pipeline in one command.

This covers:
1. Human calibration on normalized review logs
2. Baseline synthetic benchmark run
3. Calibrated synthetic benchmark run
4. Comparison summary export
"""

from memory_decay.calibration_pipeline import main


if __name__ == "__main__":
    main()
