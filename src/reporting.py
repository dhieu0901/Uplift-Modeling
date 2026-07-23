from __future__ import annotations

import numpy as np
import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Export a DataFrame as a Markdown table without requiring tabulate."""
    table = df.reset_index() if index else df.reset_index(drop=True)
    columns = [str(column) for column in table.columns]

    def format_value(value) -> str:
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "nan"
            return f"{value:.6f}"
        return str(value)

    rows = [
        [format_value(value).replace("|", "\\|") for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        if rows
        else len(columns[index])
        for index in range(len(columns))
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    header = render_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, separator, *body])
