# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotStyle:
    # plot sizing
    inner_px: int = 600
    pad_px: int = 80

    # plot base colors
    background_color: str = "#FFFFFF"
    border_color: str = "#111111"
    legend_border_color: str = "rgba(0, 0, 0, 0.25)"
    legend_background_color: str = "rgba(255, 255, 255, 0.7)"
    grid_color: str = "#CCCCCC"

    # simple overlays
    dark_gray: str = "#222222"

    # base color: steel blue
    color_1: str = "#08519C"

    # accent 1: magenta variations
    accent_1: str = "rgba(255,45,134,1)"
    accent_1_opaque: str = "rgba(255,45,134,0.85)"
    accent_1_opaque_fill: str = "rgba(255,45,134,0.18)"

    # color sequence for multi-layered plots
    theme_sequence: tuple[str, ...] = (
        "#08519C",  # deep steel blue
        "#FF6FAE",  # soft pink-magenta
        "#7BC043",  # vivid leaf green
        "#8A63D2",  # violet
        "#00A878",  # emerald green
        "#2F6DAE",  # strong blue
        "#A3C644",  # yellow-green
        "#5B3A9E",  # deep purple
    )

    # colorbar: low-saturation steel blue
    colorbar: tuple[tuple[float, str], ...] = (
        (0.0, "rgba(0, 0, 0, 0)"),
        (0.0001, "#B5CAE6"),
        (1.0, "#08519C"),
    )


PLOT_STYLE = PlotStyle()
