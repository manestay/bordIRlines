import argparse
from pathlib import Path

import scipy.stats
from gen_all_results_table import CS_COLS, HEADER, format_group, gen_table

parser = argparse.ArgumentParser()
parser.add_argument("--input_paths", "-i", type=Path, nargs="+", required=True)
parser.add_argument("--output_path", "-o", type=Path)
parser.add_argument("--print_latex", action="store_true")
parser.add_argument("--mini", action="store_true")
parser.add_argument("--with_role", action="store_true")


if "Response Countries stdev" in HEADER:
    HEADER.remove("Response Countries stdev")
STDEV_COLS = [
    "Response Countries σ",
    "KB CS σ",
    "Control CS σ",
    "Non-control CS σ",
    "Delta CS σ",
    "Delta CS unnormalized σ",
    "Consistency CS unk σ",
    "Consistency CS all σ",
    "Response Countries mean σ",
    "Response Countries σ σ",
]
HEADER.extend(STDEV_COLS)


def calc_ci(stdev, samples, conf=0.95):
    """
    Calculate the margin of error for the confidence interval.
    """
    if samples <= 1 or stdev is None:
        return None
    df = samples - 1
    t = scipy.stats.t.ppf(1 - (1 - conf) / 2, df)
    return t * (stdev / (samples**0.5))


if __name__ == "__main__":
    args = parser.parse_args()
    df = gen_table(args.input_paths)

    for col in STDEV_COLS:
        df[col] = df[col] * 100

    CI_COLS = [col[:-1] + "CI" for col in STDEV_COLS]
    for std_col, ci_col in zip(STDEV_COLS, CI_COLS):
        df[ci_col] = df[std_col].apply(lambda s: calc_ci(s, 10))

    HEADER.extend(CI_COLS)

    sort_order = ["Model", "IR Mode num", "Role"]
    if not args.with_role:
        df.drop(columns=["Role"], inplace=True)
        sort_order.remove("Role")
    df = df.sort_values(by=sort_order, ignore_index=True)
    df.drop(columns=["IR Mode num"], inplace=True)
    df.to_csv(args.output_path, index=False, float_format="%.3f")

    if args.print_latex:
        for col in CS_COLS:
            df[col] = df[col].apply(lambda x: round(x, 1))
        for col in ["Response Countries mean", "Response Countries σ"] + STDEV_COLS + CI_COLS:
            df[col] = df[col].apply(lambda x: round(x, 3))

        grouped = df.groupby("Model")

        if args.mini:
            mini_header = [
                "Model",
                "IR Mode",
                "KB CS",
                "Control CS",
                "Non-control CS",
                "Delta CS",
                "Consistency CS all",
            ]
            df = df[mini_header]
            mini_models = set(["gpt-4o", "gpt-4o-mini", "llama8b", "commandr"])
            df = df[df["IR Mode"] != "ablate_swap"]
            df = df[df["Model"].isin(mini_models)]

        formatted_df = df.groupby("Model").apply(format_group)
        formatted_df["IR Mode"] = formatted_df["IR Mode"].map(lambda x: x.replace("_", "\\_"))

        print(formatted_df.to_latex(index=False))
