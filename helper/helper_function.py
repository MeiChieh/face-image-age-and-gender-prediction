import matplotlib.pyplot as plt
from termcolor import cprint
import pandas as pd
from typing import List

# plotting related
def fig_size(w: int = 3, h: int = 3) -> None:
    """
    Lazy function for setting figure size

    Args:
        w (int, optional): set fig width. Defaults to 3.
        h (int, optional): set fig length. Defaults to 3.
    """
    plt.figure(figsize=(w, h))


def bprint(input: str) -> None:
    """
    Style printing with color

    Args:
        input (any): content to print
    """
    cprint(f"\n{input}", "green", attrs=["bold"])
    
    
def mark_bar(plot, digit=3) -> None:
    """
    Append bar values on the bars

    Args:
        plot (matplotlib axis): plot
    """
    for i in plot.containers:
        plot.bar_label(i, fmt=f"%.{digit}f")
        
def mark_df_color(col: pd.Series, id, color="rosybrown")-> List[str]:
    """
    Mark specified column or row with color for dataframe

    Args:
        col: pandas series passed in from apply method
        id: index of the column or row
        color: color for marking, default to rosybrown

    Returns:
        List[str]: list of background color styles for each cell in the column or row
    """

    def mark():
        return [
            (f"background-color: {color}" if idx == id else "background-color: ")
            for idx, _ in enumerate(col)
        ]

    return mark()



