# this has benen mostly borrowed from the qutip-tensorflow implementation
import cupy as cp
import json
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import argparse
import glob
from pathlib import Path

import benchmark_tools


def unravel(data, key):
    """Transforms {key:{another_key: values, another_key2: value2}} into
    {key_another_key:value, key_another_key2:value}"""
    for d in data:
        values = d.pop(key)
        for k, v in values.items():
            d[key + "_" + k] = v
    return data


def benchmark_to_dataframe(filepath):
    """Loads a JSON file where the benchmark is stored and returns a dataframe
    with the benchmar information."""
    with open(filepath) as f:
        data = json.load(f)
        data = data["benchmarks"]
        data = unravel(data, "options")
        data = unravel(data, "stats")
        data = unravel(data, "params")
        data = unravel(data, "extra_info")
        data = pd.DataFrame(data)

        # Set operation properly (for example: matmul instead of:
        # UNSERIALIZABLE[<function Qobj.__matmul__ at 0x...)
        # The name of the operation is obtained from the group name
        data["params_get_operation"] = data.group.str.split("-")
        data["params_get_operation"] = [d[-1] for d in data.params_get_operation]
        print(data.params_get_operation)
        return data


def plot_benchmark(df, destination_folder):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    grouped = df.groupby(["params_get_operation"])
    for operation, group in grouped:
        for dtype, g in group.groupby("extra_info_dtype"):
            plt.errorbar(
                g.params_size, g.stats_mean, g.stats_stddev, fmt=".-", label=dtype
            )

        plt.title(f"{operation}")
        plt.legend()
        plt.yscale("log")
        plt.xscale("log")
        plt.savefig(f".benchmarks/figures/{operation}.png")
        plt.xlabel("Size")
        plt.ylabel("Time (s)")
        plt.close()


def run_benchmarks(args):
    "Run pytest benchmark with sensible defaults."
    pytest.main(
        [
            "benchmarks",
            "--benchmark-only",
            "--benchmark-columns=Mean,StdDev,rounds,Iterations",
            "--benchmark-sort=name",
            "--benchmark-autosave",
            "-Wdefault",
        ]
        + args
    )


def get_latest_benchmark_path():
    """Returns the path to the latest benchmark run from `./.benchmarks/`"""

    benchmark_paths = glob.glob("./.benchmarks/*/*.json")
    dates = ["".join(_b.split("/")[-1].split("_")[2:4]) for _b in benchmark_paths]
    benchmarks = {date: value for date, value in zip(dates, benchmark_paths)}

    dates.sort()
    latest = dates[-1]
    benchmark_latest = benchmarks[latest]

    return benchmark_latest


def main(args=[]):
    parser = argparse.ArgumentParser(
        description="""Run and plot the benchmarks.
                                     The script also accepts the same arguments
                                     as pytest/pytest-benchmark. The script must be run
                                     from the root of the repository."""
    )
    parser.add_argument(
        "--save_csv",
        default=".benchmarks/latest.csv",
        help="""Path where the latest benchmark resulst will be
                        stored as csv. If empty it will not store results as
                        csv. Default: .benchmarks/latest.csv""",
    )
    parser.add_argument(
        "--save_plots",
        default=".benchmarks/figures",
        help="""Path where the plots will be saved. If empty,
                        it will not save the plots. Default:
                        .benchmarks/figures""",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="""If included, it will not run the benchmarks but
                        just plot the latest results from .benchmaks/ folder.
                        """,
    )

    parser.add_argument(
        "--device_id",
        default=0,
        help="""Device id for benchmarking.
                        """,
    )

    if args:
        args, other_args = parser.parse_known_args([])
    else:
        args, other_args = parser.parse_known_args()

    benchmark_tools._DEVICE = args.device_id

    if not args.plot_only:
        run_benchmarks(other_args)

    with cp.cuda.device.Device(benchmark_tools._DEVICE) as device:

        print("The sepcifications for your current device are:")
        print(device.attributes)

    benchmark_latest = get_latest_benchmark_path()
    benchmark_latest = benchmark_to_dataframe(benchmark_latest)

    # Save results as csv
    if args.save_csv:
        benchmark_latest.to_csv(args.save_csv)

    if args.save_plots:
        Path(args.save_plots).mkdir(parents=True, exist_ok=True)
        plot_benchmark(benchmark_latest, args.save_plots)


if __name__ == "__main__":
    main()
