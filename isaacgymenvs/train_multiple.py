""" Simple script to run sweeps"""

import os
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str)
    parser.add_argument("--sweep-param", type=str, default="")
    parser.add_argument("--sweep-vals", nargs='+', type=str, default=[])
    parser.add_argument("--print-only", default=False, action='store_true')

    args = parser.parse_args()

    for val in args.sweep_vals:
        command = f"{args.command} {args.sweep_param}={val}"
        print(command)
        if not args.print_only:
            os.system(command)