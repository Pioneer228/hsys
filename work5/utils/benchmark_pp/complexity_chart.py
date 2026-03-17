import plotly.graph_objects as go
from pathlib import Path
import dry

PATH_TO_CHART = f"complexity_chart_{dry.timestamp()}.html"


def make_complexity_chart(
    complexity: dict,
    path_to_chart=PATH_TO_CHART,
    cpu_time=False,
    width=1000,
    height=600,
    xaxis_log=True,
    yaxis_log=True,
    dark=False,
) -> go.Figure:
    fig = go.Figure()

    for benchmark, df in complexity.items():
        fig.add_trace(
            go.Scatter(
                x=df["size"],
                y=df["cpu_time"] if cpu_time else df["real_time"],
                mode="lines+markers",
                name=benchmark,
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Real Complexity",
        title_x=0.5,
        xaxis_title="N",
        yaxis_title=f"Time, {list(complexity.values())[0].time_unit[0]}",
        xaxis_type="log" if xaxis_log else "linear",
        yaxis_type="log" if yaxis_log else "linear",
        legend=dict(
            title="Benchmarks:",
            orientation="v",
            yanchor="top",
            y=-0.3,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        template="plotly_dark" if dark else "plotly_white",
        width=width,
        height=height,
        autosize=False,
    )

    fig.write_html(path_to_chart)
    print(f"The chart file has been saved to {path_to_chart}.")

    return fig


def main(argv):
    argparser = dry.make_default_argparser()

    argparser.add_argument(
        "-c",
        "--chart",
        type=str,
        default=str(PATH_TO_CHART),
        help="Output path for the chart file",
    )

    args = argparser.parse_args(argv)

    dry.show_chart(
        make_complexity_chart(
            dry.parse_complexity(Path(args.json)),
            args.chart,
            args.cpu,
            args.width,
            args.height,
            args.xlog,
            args.ylog,
            args.dark,
        ),
        args.chart,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
