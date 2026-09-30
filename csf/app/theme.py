# SPDX-License-Identifier: Apache-2.0
"""Titlebar, panel label and dark mode."""
from __future__ import annotations

TITLE = "CSF · Contextual Safety Filtering"
BRAND_COLOR = (152, 189, 255)

# Let the top-left notification stack clear the titlebar, and let notification
# bodies honour newlines (the scene legend is one entry per line).  These are
# Mantine class hashes as shipped with viser 1.0.x.
_NOTIFICATION_CSS = (
    "<style>"
    ".m_b37d9ac7{top:64px !important;}"
    ".m_3d733a3a{white-space:pre-line !important;"
    "overflow:visible !important;text-overflow:clip !important;}"
    "</style>"
)


def install_notification_css(client: object) -> None:
    try:
        client.gui.add_html(_NOTIFICATION_CSS)
    except Exception as exc:  # noqa: BLE001 - cosmetic only
        print(f"[csf] notification CSS skipped: {exc}")


def configure_theme(client: object, *, dark_mode: bool = False, grid: object | None = None,
                    titlebar_dark_mode_checkbox_uuid: str | None = None) -> None:
    """Apply the theme; the grid follows the light/dark palette."""
    from kimodo.demo.config import DARK_THEME, LIGHT_THEME  # noqa: PLC0415
    from viser.theme import TitlebarButton, TitlebarConfig  # noqa: PLC0415

    if grid is not None:
        grid.section_color = (DARK_THEME if dark_mode else LIGHT_THEME)["grid"]
    buttons = (
        TitlebarButton(text="Kimodo", icon=None, href="https://github.com/nv-tlabs/kimodo"),
        TitlebarButton(text="ARDY", icon=None, href="https://github.com/nv-tlabs/ardy"),
        TitlebarButton(text="ECHO", icon=None, href="https://github.com/Hxxxz0/ECHO_CODE"),
        TitlebarButton(text="MotionHiFlow", icon=None, href="https://github.com/ai-lh/MotionHiFlow"),
    )
    client.gui.set_panel_label("CSF")
    client.gui.configure_theme(
        titlebar_content=TitlebarConfig(buttons=buttons, image=None, title_text=TITLE),
        control_layout="floating",
        control_width="large",
        dark_mode=dark_mode,
        show_logo=False,
        show_share_button=False,
        titlebar_dark_mode_checkbox_uuid=titlebar_dark_mode_checkbox_uuid,
        brand_color=BRAND_COLOR,
    )
