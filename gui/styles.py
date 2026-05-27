import tkinter as tk
from tkinter import ttk


class StyleMixin:
    """ttk 视觉样式配置。"""

    def _setup_styles(self):
        """统一配置 ttk 视觉样式：颜色、字体、内边距，让界面更现代化。"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        # 调色板
        PRIMARY = "#1976d2"      # 主操作（推理、抓取等）
        PRIMARY_HOVER = "#1565c0"
        SUCCESS = "#2e7d32"      # 安全/确认操作（回 Home、张开）
        SUCCESS_HOVER = "#1b5e20"
        WARN = "#ef6c00"         # 警示操作（前往抓取/释放）
        WARN_HOVER = "#e65100"
        DANGER = "#c62828"       # 危险操作（自动运行/复位）
        DANGER_HOVER = "#8e0000"
        MUTED = "#546e7a"        # 次要按钮（清除、复位坐标）
        MUTED_HOVER = "#37474f"
        TEXT = "#1a1a1a"
        BG = "#f5f6f8"
        CARD_BG = "#ffffff"

        self.root.configure(bg=BG)

        # ---- 文本/容器样式 ----
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD_BG)

        style.configure("TLabel", background=BG, foreground=TEXT, font=("Microsoft YaHei", 10))
        style.configure("Title.TLabel",
                        background=BG, foreground=PRIMARY,
                        font=("Microsoft YaHei", 18, "bold"))
        style.configure("Subtitle.TLabel",
                        background=BG, foreground="#666",
                        font=("Microsoft YaHei", 10))
        style.configure("Section.TLabel",
                        background=BG, foreground=TEXT,
                        font=("Microsoft YaHei", 11, "bold"))
        style.configure("Value.TLabel",
                        background=BG, foreground="#0d47a1",
                        font=("Consolas", 10))
        style.configure("Status.TLabel",
                        background="#263238", foreground="#e0f7fa",
                        font=("Consolas", 10), padding=(8, 4))

        style.configure("TLabelframe", background=BG, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label",
                        background=BG, foreground=PRIMARY,
                        font=("Microsoft YaHei", 10, "bold"))

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                        font=("Microsoft YaHei", 10, "bold"),
                        padding=(16, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", PRIMARY)],
                  foreground=[("selected", "white")])

        # ---- 按钮样式 ----
        base_btn = dict(font=("Microsoft YaHei", 10), padding=(10, 6), borderwidth=0)
        style.configure("TButton", **base_btn)

        def _color_button(name, base, hover, fg="white"):
            style.configure(name, background=base, foreground=fg, **base_btn)
            style.map(name,
                      background=[("active", hover), ("pressed", hover)],
                      foreground=[("active", fg)])

        _color_button("Primary.TButton", PRIMARY, PRIMARY_HOVER)
        _color_button("Success.TButton", SUCCESS, SUCCESS_HOVER)
        _color_button("Warn.TButton", WARN, WARN_HOVER)
        _color_button("Danger.TButton", DANGER, DANGER_HOVER)
        _color_button("Muted.TButton", MUTED, MUTED_HOVER)
        _color_button("Accent.TButton", PRIMARY, PRIMARY_HOVER)  # 兼容旧引用

        # 复选框
        style.configure("TCheckbutton",
                        background=BG, foreground=TEXT,
                        font=("Microsoft YaHei", 10))
        # 下拉框
        style.configure("TCombobox", padding=4)
