#!/usr/bin/env python
"""
File: make_figure
Date: 5/23/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    args = parse_args()


    df = pd.read_csv(args.input_csv)

    plt.figure()
    plt.plot(df.Step, df.Value, 'lightblue')
    plt.plot(df.Step, df.Value.rolling(25).mean(), 'g')
    plt.xlabel("Gradient descent step")
    plt.ylabel("Huber loss")

    title = "Loss during training"
    plt.title(title)

    plt.savefig(f"{args.output_name}.png")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="plot some results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_csv", help="Input CSV to plot")
    parser.add_argument("output_name", help="Name for saving the plot image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
