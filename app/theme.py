"""Shared Gradio theme: flat colour, IBM Plex type, no gradients or glow.

Palette: warm paper background, near-black ink, one rust accent. Chosen to read as
editorial and technical rather than the default purple-blue "AI app" look.
"""

from __future__ import annotations

import gradio as gr

PAPER = "#f6f3ee"
SURFACE = "#fffdf9"
INK = "#1d1b18"
MUTED = "#6b665e"
RULE = "#d9d3c8"
ACCENT = "#a8441f"
ACCENT_HOVER = "#8c3818"

_FONT = (
    gr.themes.GoogleFont("IBM Plex Sans"),
    "ui-sans-serif",
    "system-ui",
    "sans-serif",
)
_MONO = (
    gr.themes.GoogleFont("IBM Plex Mono"),
    "ui-monospace",
    "SFMono-Regular",
    "monospace",
)

CSS = f"""
.gradio-container {{ max-width: 1080px !important; margin: 0 auto; }}
h1, h2, h3 {{ letter-spacing: -0.01em; font-weight: 600; }}
.notice {{
  border-left: 3px solid {ACCENT};
  background: {SURFACE};
  padding: 10px 14px;
  color: {INK};
  font-size: 0.92rem;
}}
.num, table td, table th {{ font-variant-numeric: tabular-nums; }}
footer {{ display: none !important; }}
"""


def build_theme() -> gr.themes.Base:
    """Return a flat, single-accent theme. All gradient-like fills are overridden to solid."""
    return gr.themes.Base(
        font=_FONT,
        font_mono=_MONO,
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_md,
    ).set(
        body_background_fill=PAPER,
        body_text_color=INK,
        body_text_color_subdued=MUTED,
        background_fill_primary=SURFACE,
        background_fill_secondary=PAPER,
        border_color_primary=RULE,
        block_background_fill=SURFACE,
        block_border_width="1px",
        block_shadow="none",
        block_label_text_color=MUTED,
        input_background_fill=SURFACE,
        input_border_color=RULE,
        input_shadow="none",
        input_shadow_focus="none",
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_hover=ACCENT_HOVER,
        button_primary_text_color="#ffffff",
        button_primary_border_color=ACCENT,
        button_secondary_background_fill=SURFACE,
        button_secondary_background_fill_hover=PAPER,
        button_secondary_text_color=INK,
        button_secondary_border_color=RULE,
        button_primary_shadow="none",
        button_secondary_shadow="none",
        slider_color=ACCENT,
        link_text_color=ACCENT,
        link_text_color_hover=ACCENT_HOVER,
    )
