import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_paths", "-i", type=Path, nargs="+", required=True)
parser.add_argument("--output_path", "-o", type=Path)
parser.add_argument("--print_latex", action="store_true")
parser.add_argument("--mini", action="store_true")
parser.add_argument("--with_role", action="store_true")

HEADER = [
    "Model",
    "IR Mode",
    "Role",
    "KB CS",
    "Control CS",
    "Non-control CS",
    "Delta CS",
    "Delta CS unnormalized",
    "Consistency CS unk",
    "Consistency CS all",
    "Response Countries mean",
    "Response Countries stdev",
]
CS_COLS = [col for col in HEADER if "CS" in col]
IR_MODES = ["no_ir", "qlang", "qlang_en", "en", "rel_langs", "ablate_swap"]
IR_MODE_ORDER = {mode: i for i, mode in enumerate(IR_MODES)}


def get_config(stem):
    arr = stem.split("_")
    if len(arr) == 6:  # handle IR modes `no_ir` and `rel_langs`
        arr[1] += "_" + arr[2]
        arr.pop(2)

    if len(arr) != 5:
        print("WARNING: could not parse config from stem:", stem)

        return {"Model": stem}

    role = arr[0].split("-")[1] if "-" in arr[0] else arr[2]
    return {
        "IR Mode": arr[1],
        "Role": role,
        "Model": arr[3],
        "Num Paragraphs": arr[4],
    }


def format_group(group_df):
    for col in group_df.columns:
        if group_df[col].dtype != "float":
            continue
        vals_dedup = group_df[col].drop_duplicates()
        if len(vals_dedup) < 2:
            max_val1 = max_val2 = vals_dedup.iloc[0]
        elif "Delta" not in col:
            # get top 2 largest unique values
            max_val1, max_val2 = vals_dedup.nlargest(2)
        else:  # use min for delta
            max_val1, max_val2 = vals_dedup.nsmallest(2)
        group_df[col] = group_df[col].apply(
            lambda x: f"\\textbf{{{x}}}"
            if x == max_val1
            else (f"\\underline{{{x}}}" if x == max_val2 else f"{x}")
        )
    return group_df


def gen_table(input_paths):
    rows = []
    for input_path in input_paths:
        stem = input_path.stem
        config = get_config(stem)
        with input_path.open() as f:
            data = json.load(f)
        data.update(config)

        rows.append(data)

    df = pd.DataFrame(rows, columns=HEADER)

    # round all columns with CS
    for col in CS_COLS:
        df[col] = df[col] * 100
    df["IR Mode num"] = df["IR Mode"].map(IR_MODE_ORDER)
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    df = gen_table(args.input_paths)
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
        for col in ["Response Countries mean", "Response Countries stdev"]:
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
