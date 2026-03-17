import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from argparse import ArgumentTypeError, ArgumentParser
from functools import partial


def parse_complexity(path_to_result: Path) -> dict:
    if not path_to_result.exists():
        raise FileNotFoundError("File not found")

    with open(path_to_result, "r") as f:
        data = json.load(f)

    def parse_benchmark_and_size(x: str):
        tokens = x.split("/")
        i = 0 if tokens[1].isdigit() else 1
        return tokens[i], int(tokens[i + 1])

    df = pd.DataFrame(data["benchmarks"])
    df["benchmark"] = df["name"].apply(lambda x: parse_benchmark_and_size(x)[0])
    df["size"] = df["name"].apply(lambda x: parse_benchmark_and_size(x)[1])

    complexity = {}

    for benchmark in df["benchmark"]:
        if benchmark not in complexity:
            complexity[benchmark] = df[df["benchmark"] == benchmark].sort_values(
                "size"
            )  # pyright: ignore

    if len(complexity) == 0:
        raise ValueError("Data not found")

    return complexity


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def show_chart(fig, path_to_chart):
    try:
        from google.colab import output  # noqa # pyright: ignore
        from IPython.display import display, HTML  # pyright: ignore

        display(HTML(fig.to_html()))
    except ImportError:
        import webbrowser

        webbrowser.open(path_to_chart)


def range_limited_int(min_value, max_value, value) -> int:
    ivalue = int(value)
    if not min_value <= ivalue <= max_value:
        raise ArgumentTypeError(
            f"Value must be between {min_value} and {max_value}, got {ivalue}"
        )
    return ivalue


def make_default_argparser():
    argparser = ArgumentParser()

    argparser.add_argument(
        "-j",
        "--json",
        required=True,
        type=str,
        help="Path to benchmark results (JSON)",
    )

    argparser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="CPU Time Chart",
    )

    argparser.add_argument(
        "--xlog",
        action="store_true",
        default=False,
        help="Log scale on X axis",
    )

    argparser.add_argument(
        "--ylog",
        action="store_true",
        default=False,
        help="Log scale on Y axis",
    )

    argparser.add_argument(
        "-wx",
        "--width",
        type=partial(range_limited_int, min_value=100, max_value=1920),
        default=1000,
        help="Width of the chart in pixels",
    )

    argparser.add_argument(
        "-hy",
        "--height",
        type=partial(range_limited_int, min_value=100, max_value=1080),
        default=600,
        help="Height of the chart in pixels",
    )

    argparser.add_argument(
        "--dark",
        action="store_true",
        default=False,
        help="Use dark theme",
    )

    return argparser
