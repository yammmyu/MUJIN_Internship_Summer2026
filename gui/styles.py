"""Mujin Humanoid Control Console — visual design system.

One place for the whole app's look: a light, professional "command console"
palette, a runtime-resolved font stack (so it looks right on Windows, macOS and
Linux), and every ttk style name the panels use. Panels never hard-code colors
or fonts — they read tokens from ``self.theme`` (a ``Theme`` instance the mixin
attaches) and reference the named styles below.
"""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


# --------------------------------------------------------------------------- #
#  Design tokens                                                              #
# --------------------------------------------------------------------------- #
class Theme:
    """Immutable-ish bag of design tokens, plus resolved font families.

    Colors are a calm light palette tuned to read well on a projector / camera:
    high text contrast, restrained accents, one confident brand blue.
    """

    # ---- brand + surfaces ----
    BRAND = "#2F6BED"          # Mujin blue — primary actions, brand marks
    BRAND_DARK = "#1F53C4"     # pressed / hover
    BRAND_SOFT = "#EAF1FE"     # tinted fill (selected rows, chips)
    BRAND_INK = "#173A8A"      # brand text on light

    APP_BG = "#EEF1F6"         # window background (behind the cards)
    SURFACE = "#FFFFFF"        # card background
    SURFACE_ALT = "#F7F9FC"    # zebra / inset panels
    HEADER_BG = "#FFFFFF"      # top app bar
    BORDER = "#DCE2EC"         # hairline card border
    BORDER_STRONG = "#C4CCDA"

    # ---- text ----
    TEXT = "#101828"           # primary text
    TEXT_MUTED = "#667085"     # secondary / captions
    TEXT_FAINT = "#98A2B3"     # disabled / hints
    TEXT_ON_BRAND = "#FFFFFF"

    # ---- semantic accents ----
    SUCCESS = "#12805C"
    SUCCESS_DARK = "#0B5F44"
    SUCCESS_SOFT = "#E6F4EF"
    WARN = "#B5540A"
    WARN_DARK = "#8A3F06"
    WARN_SOFT = "#FCF0E4"
    DANGER = "#C0392B"
    DANGER_DARK = "#96271B"
    DANGER_SOFT = "#FBEAE7"
    MUTED = "#5B6B7B"          # neutral secondary buttons
    MUTED_DARK = "#3E4A57"

    # ---- status dots ----
    DOT_ON = "#17A673"
    DOT_OFF = "#B7C0CC"

    # ---- spacing scale (px) ----
    PAD_XS, PAD_SM, PAD_MD, PAD_LG, PAD_XL = 4, 8, 12, 16, 24

    def __init__(self, root):
        fams = set(tkfont.families(root))

        def pick(candidates, fallback):
            for c in candidates:
                if c in fams:
                    return c
            return fallback

        # A clean humanist sans for UI, and a real monospace for numbers/telemetry.
        self.font_family = pick(
            ["Segoe UI", "Inter", "Roboto", "Helvetica Neue", "Noto Sans",
             "DejaVu Sans", "Arial"], "TkDefaultFont")
        self.mono_family = pick(
            ["Cascadia Mono", "Cascadia Code", "JetBrains Mono", "Consolas",
             "Menlo", "DejaVu Sans Mono", "Roboto Mono"], "TkFixedFont")

    # Font helpers (size, weight) -> tk font tuple.
    def ui(self, size=10, weight="normal"):
        return (self.font_family, size, weight)

    def mono(self, size=10, weight="normal"):
        return (self.mono_family, size, weight)


class StyleMixin:
    """Applies the Mujin light theme to ttk and exposes ``self.theme`` tokens."""

    def _setup_styles(self):
        self.theme = Theme(self.root)
        t = self.theme
        style = ttk.Style()
        try:
            style.theme_use("clam")     # clam honors background/foreground overrides
        except tk.TclError:
            pass

        self.root.configure(bg=t.APP_BG)

        # ---- containers ----
        style.configure("TFrame", background=t.APP_BG)
        style.configure("Card.TFrame", background=t.SURFACE,
                        relief="solid", borderwidth=1, bordercolor=t.BORDER)
        style.configure("Header.TFrame", background=t.HEADER_BG)
        style.configure("Toolbar.TFrame", background=t.SURFACE_ALT)

        # ---- text ----
        style.configure("TLabel", background=t.APP_BG, foreground=t.TEXT,
                        font=t.ui(10))
        style.configure("Title.TLabel", background=t.HEADER_BG,
                        foreground=t.TEXT, font=t.ui(17, "bold"))
        style.configure("Subtitle.TLabel", background=t.HEADER_BG,
                        foreground=t.TEXT_MUTED, font=t.ui(10))
        style.configure("Brand.TLabel", background=t.HEADER_BG,
                        foreground=t.BRAND, font=t.ui(17, "bold"))
        style.configure("Section.TLabel", background=t.APP_BG,
                        foreground=t.TEXT, font=t.ui(11, "bold"))
        style.configure("CardTitle.TLabel", background=t.APP_BG,
                        foreground=t.TEXT, font=t.ui(11, "bold"))
        style.configure("Caption.TLabel", background=t.APP_BG,
                        foreground=t.TEXT_MUTED, font=t.ui(9))
        style.configure("Value.TLabel", background=t.APP_BG,
                        foreground=t.BRAND_INK, font=t.mono(10))
        style.configure("Metric.TLabel", background=t.SURFACE,
                        foreground=t.TEXT, font=t.mono(30, "bold"))
        style.configure("MetricLabel.TLabel", background=t.SURFACE,
                        foreground=t.TEXT_MUTED, font=t.ui(9, "bold"))
        style.configure("Status.TLabel", background="#1D2939",
                        foreground="#E6EDF7", font=t.mono(10), padding=(10, 5))
        style.configure("Pill.TLabel", background=t.BRAND_SOFT,
                        foreground=t.BRAND_INK, font=t.ui(9, "bold"),
                        padding=(8, 3))
        style.configure("DemoPill.TLabel", background=t.WARN_SOFT,
                        foreground=t.WARN_DARK, font=t.ui(9, "bold"),
                        padding=(8, 3))

        # ---- labelframe = an outlined section on the grey canvas ----
        # Interior stays APP_BG so inner ttk.Frames match; the card reads via its
        # hairline border + bold title. White (SURFACE) is reserved for KPI tiles
        # and data widgets (Treeview, camera images) that should visually "pop".
        style.configure("TLabelframe", background=t.APP_BG, borderwidth=1,
                        relief="solid", bordercolor=t.BORDER, padding=t.PAD_SM)
        style.configure("TLabelframe.Label", background=t.APP_BG,
                        foreground=t.TEXT, font=t.ui(10, "bold"))

        # ---- notebook (tabs) ----
        style.configure("TNotebook", background=t.APP_BG, borderwidth=0,
                        tabmargins=(6, 6, 6, 0))
        style.configure("TNotebook.Tab", font=t.ui(10, "bold"),
                        padding=(18, 9), background=t.APP_BG,
                        foreground=t.TEXT_MUTED, borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", t.SURFACE)],
                  foreground=[("selected", t.BRAND)],
                  expand=[("selected", (0, 0, 0, 0))])

        # ---- buttons ----
        base_btn = dict(font=t.ui(10, "bold"), padding=(13, 8), borderwidth=0,
                        focusthickness=0, relief="flat")
        style.configure("TButton", background=t.SURFACE_ALT, foreground=t.TEXT,
                        **base_btn)

        def color_btn(name, base, hover, fg=t.TEXT_ON_BRAND):
            style.configure(name, background=base, foreground=fg, **base_btn)
            style.map(name,
                      background=[("pressed", hover), ("active", hover),
                                  ("disabled", "#E4E8EF")],
                      foreground=[("disabled", t.TEXT_FAINT), ("active", fg)])

        color_btn("Primary.TButton", t.BRAND, t.BRAND_DARK)
        color_btn("Success.TButton", t.SUCCESS, t.SUCCESS_DARK)
        color_btn("Warn.TButton", t.WARN, t.WARN_DARK)
        color_btn("Danger.TButton", t.DANGER, t.DANGER_DARK)
        color_btn("Muted.TButton", t.MUTED, t.MUTED_DARK)
        color_btn("Accent.TButton", t.BRAND, t.BRAND_DARK)   # legacy alias

        # Ghost / secondary outline button (light fill, brand text).
        style.configure("Ghost.TButton", background=t.SURFACE,
                        foreground=t.BRAND, **base_btn)
        style.map("Ghost.TButton",
                  background=[("pressed", t.BRAND_SOFT), ("active", t.BRAND_SOFT)],
                  foreground=[("disabled", t.TEXT_FAINT)])

        # Big, unmistakable E-stop button.
        style.configure("EStop.TButton", background=t.DANGER,
                        foreground="white", font=t.ui(12, "bold"),
                        padding=(16, 12), borderwidth=0, relief="flat")
        style.map("EStop.TButton",
                  background=[("pressed", t.DANGER_DARK), ("active", t.DANGER_DARK)])

        # ---- telemetry table (Treeview) ----
        style.configure("Monitor.Treeview", background=t.SURFACE,
                        fieldbackground=t.SURFACE, foreground=t.TEXT,
                        font=t.mono(9), rowheight=24, borderwidth=0)
        style.configure("Monitor.Treeview.Heading", background=t.SURFACE_ALT,
                        foreground=t.TEXT_MUTED, font=t.ui(9, "bold"),
                        padding=(2, 6), relief="flat")
        style.map("Monitor.Treeview.Heading",
                  background=[("active", t.BORDER)])
        style.map("Monitor.Treeview",
                  background=[("selected", t.BRAND_SOFT)],
                  foreground=[("selected", t.TEXT)])

        # ---- inputs ----
        style.configure("TCheckbutton", background=t.APP_BG, foreground=t.TEXT,
                        font=t.ui(10), focuscolor=t.APP_BG)
        style.map("TCheckbutton",
                  background=[("active", t.APP_BG)],
                  indicatorcolor=[("selected", t.BRAND), ("!selected", t.SURFACE)])
        style.configure("Card.TCheckbutton", background=t.APP_BG,
                        foreground=t.TEXT, font=t.ui(10))
        style.map("Card.TCheckbutton", background=[("active", t.APP_BG)])
        style.configure("TCombobox", padding=5, font=t.ui(10))
        style.configure("TSpinbox", padding=4, arrowsize=12,
                        fieldbackground=t.SURFACE)
        style.configure("Horizontal.TScale", background=t.SURFACE)
        style.configure("TSeparator", background=t.BORDER)

        # Scrollbar: slim and neutral.
        style.configure("Vertical.TScrollbar", background=t.BORDER_STRONG,
                        troughcolor=t.APP_BG, borderwidth=0, arrowsize=12)
        style.configure("Progress.Horizontal.TProgressbar",
                        background=t.BRAND, troughcolor=t.SURFACE_ALT,
                        borderwidth=0, thickness=10)
