try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os
import io
import base64
import argparse
import threading
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

from pipeline import process_image_path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

APP_TITLE = "Liver Ultrasound AI"
APP_SUBTITLE = "Automated image analysis  \u2022  Feature-based classification  \u2022  Research prototype"

UI_FONT = "Segoe UI"

# ----------------------------------------------------------------------
# COLOR PALETTE
# ----------------------------------------------------------------------
PAGE_BG = "#eef2f9"
SIDEBAR_BG = "#ffffff"
CARD_BG = "#ffffff"
BORDER_COLOR = "#e2e8f4"
SHADOW_COLOR = "#c7d2e6"

TEXT_DARK = "#152238"
TEXT_MUTED = "#6b7688"
TEXT_FAINT = "#8b94a7"

BLUE_PRIMARY = "#8F8178"
BLUE_PRIMARY_DARK = "#756860"
BLUE_LIGHT_BG = "#E8E1DC"
BLUE_BORDER = "#C9BDB5"

NAV_HOVER_BG = "#f2f5fb"

GREEN = "#1f9d55"
GREEN_BG = "#e5f6ea"
GREEN_TEXT = "#186b39"

RED = "#dc2626"
RED_BG = "#fdecec"
RED_TEXT = "#a3221c"

AMBER = "#c9820a"
AMBER_BG = "#fdf3e0"
AMBER_TEXT = "#8a5a06"

NEUTRAL_BG = "#f1f3f7"
NEUTRAL_TEXT = "#4b5566"

IMAGE_CANVAS_BG = "#11141a"

VERDICT_STYLES = {
    "Normal":    {"bg": GREEN_BG, "fg": GREEN_TEXT, "icon_bg": GREEN, "icon": "\u2713"},
    "Benign":    {"bg": AMBER_BG, "fg": AMBER_TEXT, "icon_bg": AMBER, "icon": "!"},
    "Malignant": {"bg": RED_BG,   "fg": RED_TEXT,   "icon_bg": RED,   "icon": "\u2715"},
}

BAR_COLORS = {
    "Normal": BLUE_PRIMARY,
    "Malignant": "#5b6472",
    "Benign": "#9aa3b2",
}

LABEL_COLORS_BGR = {
    "Malignant": (40, 40, 220),
    "Benign": (30, 130, 220),
    "Normal": (70, 170, 50),
}

FEATURE_LABELS = {
    "liver_mean_intensity": "Liver mean intensity",
    "liver_std_intensity": "Liver intensity variation",
    "liver_glcm_contrast": "Liver GLCM contrast",
    "liver_glcm_energy": "Liver GLCM energy",
    "liver_glcm_homogeneity": "Liver GLCM homogeneity",
    "liver_edge_density": "Liver edge density",
    "mass_detected": "Candidate detected",
    "mass_relative_area": "Candidate relative area",
    "mass_mean_intensity": "Candidate mean intensity",
    "mass_std_intensity": "Candidate intensity variation",
    "mass_intensity_ratio": "Candidate/liver intensity ratio",
    "mass_liver_contrast": "Candidate/liver contrast",
    "mass_glcm_contrast": "Candidate GLCM contrast",
    "mass_glcm_energy": "Candidate GLCM energy",
    "mass_glcm_homogeneity": "Candidate GLCM homogeneity",
    "mass_circularity": "Candidate circularity",
    "mass_extent": "Candidate extent",
    "mass_solidity": "Candidate solidity",
    "mass_aspect_ratio": "Candidate aspect ratio",
    "mass_edge_density": "Candidate edge density",
}

PERCENT_FEATURES = {
    "liver_edge_density",
    "mass_relative_area",
    "mass_extent",
    "mass_solidity",
    "mass_edge_density",
}


# ========================================================================
# DRAWING HELPERS
# ========================================================================
def rounded_rect(canvas, x1, y1, x2, y2, radius=10, fill="", outline="", width=1, tags=None, **kwargs):
    """Draw a crisp, clean rounded rectangle using exact geometric primitives
    (arcs, lines, and rectangles) with zero fuzzy/smoothed polygon spline artifacts."""
    tag = tags or kwargs.get("tags") or kwargs.get("tag")
    tag_kw = {"tags": tag} if tag else {}
    radius = max(0, min(int(radius), int(abs(x2 - x1) // 2), int(abs(y2 - y1) // 2)))
    d = 2 * radius
    items = []

    if radius <= 0:
        return [canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=width, **tag_kw)]

    # 1. Fill interior using 3 overlapping rects + 4 corner pieslices
    if fill and fill != "":
        items.append(canvas.create_rectangle(x1 + radius, y1, x2 - radius, y2, fill=fill, outline=fill, **tag_kw))
        items.append(canvas.create_rectangle(x1, y1 + radius, x1 + radius, y2 - radius, fill=fill, outline=fill, **tag_kw))
        items.append(canvas.create_rectangle(x2 - radius, y1 + radius, x2, y2 - radius, fill=fill, outline=fill, **tag_kw))
        items.append(canvas.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, fill=fill, outline=fill, style="pieslice", **tag_kw))
        items.append(canvas.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, fill=fill, outline=fill, style="pieslice", **tag_kw))
        items.append(canvas.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, fill=fill, outline=fill, style="pieslice", **tag_kw))
        items.append(canvas.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, fill=fill, outline=fill, style="pieslice", **tag_kw))

    # 2. Draw border outline using 4 straight lines + 4 corner arcs
    if outline and outline != "":
        w = max(1, int(width))
        items.append(canvas.create_line(x1 + radius, y1, x2 - radius, y1, fill=outline, width=w, **tag_kw))
        items.append(canvas.create_line(x1 + radius, y2, x2 - radius, y2, fill=outline, width=w, **tag_kw))
        items.append(canvas.create_line(x1, y1 + radius, x1, y2 - radius, fill=outline, width=w, **tag_kw))
        items.append(canvas.create_line(x2, y1 + radius, x2, y2 - radius, fill=outline, width=w, **tag_kw))
        items.append(canvas.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, outline=outline, width=w, style="arc", **tag_kw))
        items.append(canvas.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, outline=outline, width=w, style="arc", **tag_kw))
        items.append(canvas.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, outline=outline, width=w, style="arc", **tag_kw))
        items.append(canvas.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, outline=outline, width=w, style="arc", **tag_kw))

    return items


class Card(tk.Frame):
    """A clean, crisp modern white card container with a 1px hairline border.
    Eliminates all canvas drop-shadow fuzziness and redraw overhead.
    Place any widgets inside `.body`."""

    def __init__(self, parent, pad=18, radius=16, shadow=True,
                 card_bg=CARD_BG, page_bg=PAGE_BG, border=BORDER_COLOR):
        super().__init__(parent, bg=page_bg)
        self.pad = pad
        self.radius = radius
        self.shadow = shadow
        self.card_bg = card_bg
        self.page_bg = page_bg
        self.border = border

        self.card_frame = tk.Frame(
            self,
            bg=card_bg,
            highlightthickness=1,
            highlightbackground=border,
            bd=0,
        )
        self.card_frame.pack(fill="both", expand=True, padx=1, pady=1)

        self.body = tk.Frame(self.card_frame, bg=card_bg)
        self.body.pack(fill="both", expand=True, padx=pad, pady=pad)

    def pack_propagate(self, flag):
        super().pack_propagate(flag)
        self.card_frame.pack_propagate(flag)

    def grid_propagate(self, flag):
        super().grid_propagate(flag)
        self.card_frame.grid_propagate(flag)

    def configure(self, cnf=None, **kw):
        if "height" in kw:
            self.card_frame.configure(height=kw["height"])
        return super().configure(cnf, **kw)


class NavButton(tk.Frame):
    """Sidebar navigation pill button with consistent icon-text spacing and vertical centering."""

    def __init__(self, parent, icon, text, command, width=190, height=46):
        super().__init__(parent, bg=SIDEBAR_BG)
        self.command = command
        self.active = False
        self.width = width
        self.height = height
        self.icon = icon
        self.text = text

        self.canvas = tk.Canvas(self, width=width, height=height,
                                 bg=SIDEBAR_BG, highlightthickness=0, bd=0)
        self.canvas.pack()
        self._draw()

        self.canvas.bind("<Button-1>", lambda e: self.command())
        self.canvas.bind("<Enter>", lambda e: self._draw(hover=True))
        self.canvas.bind("<Leave>", lambda e: self._draw(hover=False))

    def _draw(self, hover=False):
        self.canvas.delete("all")
        if self.active:
            rounded_rect(self.canvas, 0, 0, self.width, self.height, 10,
                         fill=BLUE_PRIMARY, outline="")
            fg = "#ffffff"
            weight = "bold"
        elif hover:
            rounded_rect(self.canvas, 0, 0, self.width, self.height, 10,
                         fill=NAV_HOVER_BG, outline="")
            fg = TEXT_DARK
            weight = "normal"
        else:
            fg = TEXT_MUTED
            weight = "normal"

        f_icon = tkfont.Font(family="Segoe UI Symbol", size=13)
        # Dedicated icon column centered at x=28
        self.canvas.create_text(30, self.height / 2, text=self.icon,
                                 font=f_icon, fill=fg, anchor="center")
        # Text starts at x=54, ensuring 15-18px clear horizontal gap across all nav buttons
        self.canvas.create_text(62, self.height / 2, text=self.text,
                                 font=(UI_FONT, 11, weight), fill=fg, anchor="w")

    def set_active(self, active):
        self.active = active
        self._draw()


class PillButton(tk.Frame):
    """Rounded, clickable button drawn on a canvas with real rounded corners,
    exact font-metric measurement, and guaranteed horizontal clearance between icon and text."""

    def __init__(self, parent, text, command, bg=BLUE_PRIMARY, fg="#ffffff",
                 hover_bg=None, border=None, icon=None, width=140, height=38,
                 font_size=10, page_bg=CARD_BG):
        super().__init__(parent, bg=page_bg)
        self.command = command
        self.bg = bg
        self.fg = fg
        self.hover_bg = hover_bg or bg
        self.border = border
        self.icon = icon
        self.text = text
        self.width = width
        self.height = height
        self.font_size = font_size
        self.enabled = True

        self.canvas = tk.Canvas(self, width=width, height=height,
                                 bg=page_bg, highlightthickness=0, bd=0)
        self.canvas.pack()
        self._draw()

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Enter>", lambda e: self._draw(hover=True))
        self.canvas.bind("<Leave>", lambda e: self._draw(hover=False))

    def _on_click(self, event=None):
        if self.enabled and self.command:
            self.command()

    def _draw(self, hover=False):
        self.canvas.delete("all")
        fill = self.bg if not self.enabled else (self.hover_bg if hover else self.bg)
        outline = self.border if self.border else ""
        fg = self.fg if self.enabled else TEXT_FAINT
        if not self.enabled:
            fill = NEUTRAL_BG
            outline = BORDER_COLOR
        rounded_rect(self.canvas, 1, 1, self.width - 1, self.height - 1, 8,
                     fill=fill, outline=outline, width=1)

        f_text = tkfont.Font(family=UI_FONT, size=self.font_size, weight="bold")
        f_icon = tkfont.Font(family="Segoe UI Symbol", size=self.font_size + 1)

        if self.icon:
            # Measure actual rendered glyph bounding boxes on canvas for 100% accuracy
            t_icon = self.canvas.create_text(-1000, -1000, text=self.icon, font=f_icon, anchor="w")
            ibox = self.canvas.bbox(t_icon)
            self.canvas.delete(t_icon)
            iw = (ibox[2] - ibox[0]) if ibox else f_icon.measure(self.icon)

            t_txt = self.canvas.create_text(-1000, -1000, text=self.text, font=f_text, anchor="w")
            tbox = self.canvas.bbox(t_txt)
            self.canvas.delete(t_txt)
            tw = (tbox[2] - tbox[0]) if tbox else f_text.measure(self.text)

            gap = 12
            total_w = iw + gap + tw
            start_x = max(12, (self.width - total_w) / 2)
            self.canvas.create_text(start_x, self.height / 2, text=self.icon,
                                     font=f_icon, fill=fg, anchor="w")
            self.canvas.create_text(start_x + iw + gap, self.height / 2, text=self.text,
                                     font=f_text, fill=fg, anchor="w")
        else:
            self.canvas.create_text(self.width / 2, self.height / 2, text=self.text,
                                     font=f_text, fill=fg, anchor="center")

    def set_enabled(self, enabled):
        self.enabled = enabled
        self._draw()


class ProbabilityBar(tk.Frame):
    """A single labeled horizontal probability bar."""

    def __init__(self, parent, label, value, color, width=420, height=20):
        super().__init__(parent, bg=CARD_BG)
        self.columnconfigure(1, weight=1)

        tk.Label(self, text=label, bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 10), width=12, anchor="w").grid(row=0, column=0, sticky="w")

        track = tk.Canvas(self, height=height, bg=CARD_BG, highlightthickness=0, bd=0)
        track.grid(row=0, column=1, sticky="ew", padx=(8, 10))

        def draw(event=None):
            track.delete("all")
            w = max(track.winfo_width(), 40)
            bar_r = (height - 4) // 2
            rounded_rect(track, 0, 2, w, height - 2, bar_r,
                         fill=NEUTRAL_BG, outline="")
            fill_w = max(int(w * value), 4) if value > 0 else 0
            if fill_w:
                rounded_rect(track, 0, 2, fill_w, height - 2, bar_r,
                             fill=color, outline="")

        track.bind("<Configure>", draw)

        tk.Label(self, text=f"{value * 100:.1f}%", bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 11, "bold"), width=6, anchor="e").grid(row=0, column=2, sticky="e")


class VerdictIcon(tk.Canvas):
    def __init__(self, parent, glyph, color, size=44):
        super().__init__(parent, width=size, height=size, bg=CARD_BG,
                          highlightthickness=0, bd=0)
        self.create_oval(2, 2, size - 2, size - 2, fill=color, outline=color)
        self.create_text(size / 2, size / 2, text=glyph, fill="#ffffff",
                         font=(UI_FONT, int(size * 0.42), "bold"))


class PillTabs(tk.Frame):
    """A simple pill-style tab bar. Calls on_change(index) when a tab is
    clicked and manages the active visual state."""

    def __init__(self, parent, labels, on_change):
        super().__init__(parent, bg=CARD_BG)
        self.on_change = on_change
        self.buttons = []
        self.active_index = 0

        row = tk.Frame(self, bg=CARD_BG)
        row.pack(fill="x")

        for i, label in enumerate(labels):
            btn = tk.Frame(row, bg=CARD_BG)
            btn.pack(side="left", padx=(0, 6))
            canvas = tk.Canvas(btn, width=max(135, 12 * len(label)), height=40,
                                bg=CARD_BG, highlightthickness=0, bd=0)
            canvas.pack()
            canvas.bind("<Button-1>", lambda e, idx=i: self.select(idx))
            self.buttons.append((canvas, label))

        underline = tk.Frame(self, bg=BORDER_COLOR, height=1)
        underline.pack(fill="x", pady=(0, 0))

        self.select(0, fire=False)

    def select(self, index, fire=True):
        self.active_index = index
        for i, (canvas, label) in enumerate(self.buttons):
            canvas.delete("all")
            w = int(canvas["width"])
            active = (i == index)
            if active:
                rounded_rect(canvas, 2, 2, w - 2, 34, 8, fill=BLUE_LIGHT_BG, outline="")
                canvas.create_line(6, 38, w - 6, 38, fill=BLUE_PRIMARY, width=2)
                fg = BLUE_PRIMARY
            else:
                fg = TEXT_MUTED
            canvas.create_text(w / 2, 19, text=label, font=(UI_FONT, 10, "bold" if active else "normal"), fill=fg)
        if fire and self.on_change:
            self.on_change(index)


# ========================================================================
# ENLARGED IMAGE VIEWER (RADIOLOGIST INSPECTION)
# ========================================================================
class ImageViewerDialog(tk.Toplevel):
    """Enlarged image inspection viewer for radiologists and clinical reviewers.
    Provides high-quality Lanczos zoom, drag-to-pan, fit-to-window, and 1:1 view
    without modifying or reloading the original source image."""

    def __init__(self, parent, pil_image, title="Ultrasound Inspection Viewer"):
        super().__init__(parent)
        self.title(title)
        self.configure(bg="#0d1017")

        # Keep original high-resolution PIL image as pristine source
        self.orig_image = pil_image.copy()
        self.orig_w, self.orig_h = self.orig_image.size

        # Window geometry centered on screen
        self._setup_window_geometry()

        # State
        self.scale = 1.0
        self.fit_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.photo = None
        self.img_id = None
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._is_dragging = False
        self._initialized = False

        # Build interface
        self._build_toolbar(title)
        self._build_canvas()
        self._bind_events()

        # Lift and focus
        self.lift()
        self.focus_set()

    def _setup_window_geometry(self):
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(1120, max(860, int(screen_w * 0.75)))
        win_h = min(840, max(640, int(screen_h * 0.75)))
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.minsize(700, 500)

    def _build_toolbar(self, title):
        self.toolbar = tk.Frame(self, bg="#161b22", height=44,
                                highlightbackground="#2d333b", highlightthickness=1)
        self.toolbar.pack(side="top", fill="x")
        self.toolbar.pack_propagate(False)

        # Left: Title and dimensions info
        left_box = tk.Frame(self.toolbar, bg="#161b22")
        left_box.pack(side="left", padx=(14, 10), fill="y")

        tk.Label(left_box, text=title, bg="#161b22", fg="#f0f6fc",
                 font=(UI_FONT, 10, "bold")).pack(side="left")
        tk.Label(left_box, text="•", bg="#161b22", fg="#6e7681",
                 font=(UI_FONT, 10)).pack(side="left", padx=8)
        tk.Label(left_box, text=f"{self.orig_w} × {self.orig_h} px", bg="#161b22",
                 fg="#8b949e", font=(UI_FONT, 9)).pack(side="left")

        # Right: Close button
        close_btn = tk.Button(
            self.toolbar, text="✕ Close", command=self.destroy,
            bg="#21262d", fg="#f0f6fc", activebackground="#da3633", activeforeground="#ffffff",
            relief="flat", bd=0, font=(UI_FONT, 9, "bold"),
            padx=12, pady=4, cursor="hand2"
        )
        close_btn.pack(side="right", padx=(10, 14), pady=7)

        # Action controls
        ctrl_box = tk.Frame(self.toolbar, bg="#161b22")
        ctrl_box.pack(side="right", padx=10, fill="y")

        tk.Label(ctrl_box, text="Scroll: Zoom  •  Drag: Pan", bg="#161b22",
                 fg="#6e7681", font=(UI_FONT, 8, "italic")).pack(side="left", padx=(0, 14))

        btn_out = tk.Button(
            ctrl_box, text=" – ", command=self._zoom_out,
            bg="#21262d", fg="#f0f6fc", activebackground="#30363d", activeforeground="#ffffff",
            relief="flat", bd=0, font=(UI_FONT, 10, "bold"),
            padx=8, pady=3, cursor="hand2"
        )
        btn_out.pack(side="left", padx=2, pady=7)

        self.zoom_var = tk.StringVar(value="100%")
        self.zoom_lbl = tk.Label(
            ctrl_box, textvariable=self.zoom_var, bg="#161b22", fg="#f0f6fc",
            font=(UI_FONT, 9, "bold"), width=6, anchor="center"
        )
        self.zoom_lbl.pack(side="left", padx=2)

        btn_in = tk.Button(
            ctrl_box, text=" + ", command=self._zoom_in,
            bg="#21262d", fg="#f0f6fc", activebackground="#30363d", activeforeground="#ffffff",
            relief="flat", bd=0, font=(UI_FONT, 10, "bold"),
            padx=8, pady=3, cursor="hand2"
        )
        btn_in.pack(side="left", padx=2, pady=7)

        btn_fit = tk.Button(
            ctrl_box, text="Fit", command=self._fit_to_window,
            bg="#21262d", fg="#f0f6fc", activebackground="#30363d", activeforeground="#ffffff",
            relief="flat", bd=0, font=(UI_FONT, 9),
            padx=10, pady=4, cursor="hand2"
        )
        btn_fit.pack(side="left", padx=(8, 2), pady=7)

        btn_100 = tk.Button(
            ctrl_box, text="100%", command=self._actual_size,
            bg="#21262d", fg="#f0f6fc", activebackground="#30363d", activeforeground="#ffffff",
            relief="flat", bd=0, font=(UI_FONT, 9),
            padx=10, pady=4, cursor="hand2"
        )
        btn_100.pack(side="left", padx=2, pady=7)

    def _build_canvas(self):
        self.canvas_frame = tk.Frame(self, bg="#0d1017")
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame, bg="#0d1017", highlightthickness=0, bd=0,
            cursor="crosshair"
        )
        self.canvas.pack(fill="both", expand=True)

    def _bind_events(self):
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Double-Button-1>", lambda e: self._fit_to_window())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.15, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1.0 / 1.15, e.x, e.y))

        # Keyboard shortcuts
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("q", lambda e: self.destroy())
        self.bind("Q", lambda e: self.destroy())
        self.bind("<plus>", lambda e: self._zoom_in())
        self.bind("<equal>", lambda e: self._zoom_in())
        self.bind("<minus>", lambda e: self._zoom_out())
        self.bind("f", lambda e: self._fit_to_window())
        self.bind("F", lambda e: self._fit_to_window())
        self.bind("0", lambda e: self._fit_to_window())
        self.bind("1", lambda e: self._actual_size())

    def _on_canvas_configure(self, event):
        if not self._initialized:
            self._initialized = True
            self._fit_to_window()

    def _fit_to_window(self):
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        margin = 32
        scale_x = (cw - margin) / self.orig_w
        scale_y = (ch - margin) / self.orig_h
        self.fit_scale = max(0.01, min(scale_x, scale_y))
        self.scale = self.fit_scale
        self.pan_x = cw / 2.0
        self.pan_y = ch / 2.0
        self._render()

    def _actual_size(self):
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.scale = 1.0
        self.pan_x = cw / 2.0
        self.pan_y = ch / 2.0
        self._render()

    def _zoom_in(self):
        self._zoom(1.2)

    def _zoom_out(self):
        self._zoom(1.0 / 1.2)

    def _on_mouse_wheel(self, event):
        factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        self._zoom(factor, center_x=event.x, center_y=event.y)

    def _zoom(self, factor, center_x=None, center_y=None):
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        if center_x is None:
            center_x = cw / 2.0
        if center_y is None:
            center_y = ch / 2.0

        old_scale = self.scale
        new_scale = old_scale * factor
        min_scale = max(0.05, self.fit_scale * 0.25)
        max_scale = max(10.0, self.fit_scale * 10.0)
        new_scale = max(min_scale, min(max_scale, new_scale))
        if abs(new_scale - old_scale) < 1e-6:
            return

        img_x = (center_x - self.pan_x) / old_scale
        img_y = (center_y - self.pan_y) / old_scale

        self.scale = new_scale
        self.pan_x = center_x - img_x * self.scale
        self.pan_y = center_y - img_y * self.scale

        self._render()

    def _render(self):
        target_w = max(1, int(round(self.orig_w * self.scale)))
        target_h = max(1, int(round(self.orig_h * self.scale)))

        resample = getattr(Image, "Resampling", Image).LANCZOS
        resized = self.orig_image.resize((target_w, target_h), resample)
        self.photo = ImageTk.PhotoImage(resized)

        if self.img_id is None:
            self.img_id = self.canvas.create_image(
                int(round(self.pan_x)), int(round(self.pan_y)),
                image=self.photo, anchor="center"
            )
        else:
            self.canvas.itemconfig(self.img_id, image=self.photo)
            self.canvas.coords(self.img_id, int(round(self.pan_x)), int(round(self.pan_y)))

        pct = int(round(self.scale * 100))
        self.zoom_var.set(f"{pct}%")

    def _on_mouse_press(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._is_dragging = True
        self.canvas.configure(cursor="fleur")

    def _on_mouse_drag(self, event):
        if not self._is_dragging:
            return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.pan_x += dx
        self.pan_y += dy
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        if self.img_id is not None:
            self.canvas.coords(self.img_id, int(round(self.pan_x)), int(round(self.pan_y)))

    def _on_mouse_release(self, event):
        self._is_dragging = False
        self.canvas.configure(cursor="crosshair")


def open_image_viewer(image, title="Ultrasound Inspection Viewer", parent=None):
    """Open an enlarged, zoomable, draggable viewer window for a PIL Image."""
    if image is None:
        return None
    return ImageViewerDialog(parent, image, title)


# ========================================================================
# MAIN APPLICATION
# ========================================================================
class LiverClassifierUI:

    def __init__(self, root, model_dir, features_path):
        self.root = root
        self.model_dir = model_dir
        self.features_path = features_path

        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_name = "Unknown"
        self.normal_stats = {}
        self.training_metadata = None

        self.current_image_path = None
        self.original_pil = None
        self.current_result = None
        self.original_display = None
        self.annotated_display = None
        self.annotated_display_photo = None

        self.report_history = []  # list of dicts for the Reports page

        self._setup_style()
        self._build_ui()
        self._load_model_and_reference()

    # ------------------------------------------------------------------
    # STYLE
    # ------------------------------------------------------------------
    def _setup_style(self):
        self.root.title(f"{APP_TITLE} \u2014 Analysis Dashboard")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 720)
        self.root.configure(bg=PAGE_BG)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=PAGE_BG)
        style.configure("Card.TFrame", background=CARD_BG)

        style.configure("Treeview",
                        font=(UI_FONT, 9),
                        rowheight=30,
                        background=CARD_BG,
                        fieldbackground=CARD_BG,
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        font=(UI_FONT, 9, "bold"),
                        background=BLUE_LIGHT_BG,
                        foreground=TEXT_DARK,
                        relief="flat")
        style.map("Treeview.Heading", background=[("active", BLUE_LIGHT_BG)])
        style.map("Treeview", background=[("selected", BLUE_LIGHT_BG)],
                  foreground=[("selected", TEXT_DARK)])

        style.layout("Clean.Vertical.TScrollbar", style.layout("Vertical.TScrollbar"))
        style.configure("Clean.Vertical.TScrollbar", background=BORDER_COLOR,
                        troughcolor=CARD_BG, bordercolor=CARD_BG, arrowsize=12)

    # ------------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------------
    def _build_ui(self):
        root_frame = tk.Frame(self.root, bg=PAGE_BG)
        root_frame.pack(fill="both", expand=True)

        self._build_sidebar(root_frame)

        content_wrap = tk.Frame(root_frame, bg=PAGE_BG)
        content_wrap.pack(side="left", fill="both", expand=True)

        self._build_header(content_wrap)

        self.pages_container = tk.Frame(content_wrap, bg=PAGE_BG)
        self.pages_container.pack(fill="both", expand=True, padx=18, pady=(4, 8))
        self.pages_container.columnconfigure(0, weight=1)
        self.pages_container.rowconfigure(0, weight=1)

        self.page_home = tk.Frame(self.pages_container, bg=PAGE_BG)
        self.page_reports = tk.Frame(self.pages_container, bg=PAGE_BG)
        self.page_about = tk.Frame(self.pages_container, bg=PAGE_BG)

        for page in (self.page_home, self.page_reports, self.page_about):
            page.grid(row=0, column=0, sticky="nsew")

        self._build_home_page(self.page_home)
        self._build_reports_page(self.page_reports)
        self._build_about_page(self.page_about)

        self._build_footer(content_wrap)

        self.show_page("home")

    # ---- Sidebar -------------------------------------------------------
    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=SIDEBAR_BG, width=240)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        logo_row = tk.Frame(sidebar, bg=SIDEBAR_BG)
        logo_row.pack(fill="x", padx=20, pady=(24, 28))

        logo_canvas = tk.Canvas(logo_row, width=40, height=40, bg=SIDEBAR_BG,
                                 highlightthickness=0, bd=0)
        logo_canvas.pack(side="left")
        rounded_rect(logo_canvas, 0, 0, 40, 40, 10, fill=BLUE_PRIMARY, outline=BLUE_PRIMARY)
        logo_canvas.create_text(20, 20, text="\u2695", fill="#ffffff", font=(UI_FONT, 17, "bold"))

        tk.Label(logo_row, text=APP_TITLE, bg=SIDEBAR_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 11, "bold"), justify="left", wraplength=155,
                 anchor="w").pack(side="left", padx=(12, 0))

        nav_wrap = tk.Frame(sidebar, bg=SIDEBAR_BG)
        nav_wrap.pack(fill="x", padx=20)

        self.nav_buttons = {}
        self.nav_buttons["home"] = NavButton(nav_wrap, "\U0001F3E0", "Home",
                                              lambda: self.show_page("home"))
        self.nav_buttons["home"].pack(fill="x", pady=5)

        self.nav_buttons["reports"] = NavButton(nav_wrap, "\U0001F4C4", "Reports",
                                                 lambda: self.show_page("reports"))
        self.nav_buttons["reports"].pack(fill="x", pady=5)

        self.nav_buttons["about"] = NavButton(nav_wrap, "\u24D8", "About",
                                               lambda: self.show_page("about"))
        self.nav_buttons["about"].pack(fill="x", pady=5)

        # Bottom promo card
        promo_wrap = tk.Frame(sidebar, bg=SIDEBAR_BG)
        promo_wrap.pack(side="bottom", fill="x", padx=16, pady=18)

        promo_canvas = tk.Canvas(promo_wrap, height=90, bg=SIDEBAR_BG,
                                  highlightthickness=0, bd=0)
        promo_canvas.pack(fill="x")

        def draw_promo(event=None):
            promo_canvas.delete("all")
            w = promo_canvas.winfo_width()
            if w < 10:
                w = 190
            rounded_rect(promo_canvas, 0, 0, w, 88, 12, fill=BLUE_PRIMARY, outline="")

            cy=promo_canvas.winfo_height() / 2
            promo_canvas.create_text(14, 40, text="AI-Powered Liver", fill="#ffffff",
                                     font=(UI_FONT, 10, "bold"), anchor="w")
            promo_canvas.create_text(14, 63, text="Ultrasound Analysis", fill="#ffffff",
                                     font=(UI_FONT, 10, "bold"), anchor="w")

        promo_canvas.bind("<Configure>", draw_promo)

    # ---- Header ---------------------------------------------------------
    def _build_header(self, parent):
        header_wrap = tk.Frame(parent, bg=PAGE_BG)
        header_wrap.pack(fill="x", padx=18, pady=(12, 0))

        header_card = Card(header_wrap, pad=12, radius=12, shadow=True)
        header_card.pack(fill="x")
        header_card.configure(height=56)
        header_card.pack_propagate(False)

        tk.Label(header_card.body, text=APP_TITLE, bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 16, "bold"), anchor="w").pack(side="left", fill="y")

    # ---- Footer -----------------------------------------------------------
    def _build_footer(self, parent):
        tk.Label(
            parent,
            text=("Research prototype \u2014 automated screening output is not a "
                  "medical diagnosis. Always confirm with a qualified professional."),
            bg=PAGE_BG, fg=TEXT_FAINT, font=(UI_FONT, 8, "italic")
        ).pack(fill="x", padx=24, pady=(0, 10))

    # ------------------------------------------------------------------
    # HOME PAGE
    # ------------------------------------------------------------------
    def _build_home_page(self, parent):
        parent.columnconfigure(0, weight=10)
        parent.columnconfigure(1, weight=11)
        parent.rowconfigure(0, weight=1)

        left = tk.Frame(parent, bg=PAGE_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=6)
        left.rowconfigure(1, weight=5)
        left.columnconfigure(0, weight=1)

        self._build_input_card(left)
        self._build_output_card(left)

        right = tk.Frame(parent, bg=PAGE_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_findings_card(right)

    # ---- Card 1: Input Image -------------------------------------------
    def _build_input_card(self, parent):
        card = Card(parent, pad=18)
        card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        body = card.body
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        self._card_heading(body, "1", "Input Image",
                           "Upload a liver ultrasound image for analysis").grid(
            row=0, column=0, sticky="ew", pady=(0, 10))

        content = tk.Frame(body, bg=CARD_BG)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        img_frame = tk.Frame(content, bg=NEUTRAL_BG, highlightthickness=1,
                             highlightbackground=BORDER_COLOR, bd=0)
        img_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.input_canvas = tk.Canvas(img_frame, bg=NEUTRAL_BG, highlightthickness=0, bd=0)
        self.input_canvas.pack(fill="both", expand=True)
        self.input_canvas.create_text(
            10, 10, text="No image selected", fill=TEXT_FAINT,
            font=(UI_FONT, 9), anchor="nw", tags="placeholder"
        )

        meta = tk.Frame(content, bg=CARD_BG)
        meta.grid(row=0, column=1, sticky="new")

        meta_row = tk.Frame(meta, bg=CARD_BG)
        meta_row.pack(fill="x", anchor="w")
        meta_icon_canvas = tk.Canvas(meta_row, width=28, height=28, bg=CARD_BG,
                                     highlightthickness=0, bd=0)
        meta_icon_canvas.pack(side="left")
        rounded_rect(meta_icon_canvas, 0, 0, 28, 28, 6, fill=BLUE_LIGHT_BG, outline="")
        meta_icon_canvas.create_text(14, 14, text="▼", font=("Segoe UI Symbol", 11),
                                     fill=BLUE_PRIMARY, anchor="center")

        tk.Label(meta_row, text="Image Details", bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 10, "bold"), anchor="w").pack(side="left", padx=(10, 0))

        self.file_name_var = tk.StringVar(value="No file selected")
        tk.Label(meta, textvariable=self.file_name_var, bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 10, "bold"), anchor="w", justify="left",
                 wraplength=190).pack(fill="x", anchor="w", pady=(10, 2))

        self.file_dims_var = tk.StringVar(value="")
        tk.Label(meta, textvariable=self.file_dims_var, bg=CARD_BG, fg=TEXT_MUTED,
                 font=(UI_FONT, 9), anchor="w").pack(fill="x", anchor="w")

        self.file_format_var = tk.StringVar(value="")
        tk.Label(meta, textvariable=self.file_format_var, bg=CARD_BG, fg=TEXT_MUTED,
                 font=(UI_FONT, 9), anchor="w").pack(fill="x", anchor="w", pady=(0, 10))

        self.status_chip_frame = tk.Frame(meta, bg=CARD_BG)
        self.status_chip_frame.pack(fill="x", anchor="w")
        self._render_status_chip("No image loaded yet", ok=None)

        buttons_row = tk.Frame(body, bg=CARD_BG)
        buttons_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        self.choose_button = PillButton(
            buttons_row, "Choose Image", self.choose_image,
            bg="#ffffff", fg=TEXT_DARK, hover_bg=NAV_HOVER_BG, border=BORDER_COLOR,
            icon="\U0001F4C1", width=170, height=38
        )
        self.choose_button.pack(side="left")

        self.analyze_button = PillButton(
            buttons_row, "Analyze", self.analyze,
            bg=BLUE_PRIMARY, fg="#ffffff", hover_bg=BLUE_PRIMARY_DARK,
            icon="\u25B6", width=140, height=38
        )
        self.analyze_button.pack(side="left", padx=(12, 0))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(buttons_row, textvariable=self.status_var, bg=CARD_BG,
                 fg=TEXT_MUTED, font=(UI_FONT, 9)).pack(side="right")

        self.input_canvas.bind("<Configure>", lambda e: self._redraw_current_images())
        self.input_canvas.bind("<Button-1>", lambda e: self._on_click_input_image())

    def _render_status_chip(self, text, ok=True):
        for child in self.status_chip_frame.winfo_children():
            child.destroy()
        if ok is None:
            fg, glyph = TEXT_FAINT, "\u25CF"
        elif ok:
            fg, glyph = GREEN_TEXT, "\u2713"
        else:
            fg, glyph = RED_TEXT, "\u2715"
        row = tk.Frame(self.status_chip_frame, bg=CARD_BG)
        row.pack(anchor="w")
        tk.Label(row, text=glyph, bg=CARD_BG, fg=fg, font=(UI_FONT, 9, "bold")).pack(side="left")
        tk.Label(row, text=text, bg=CARD_BG, fg=fg, font=(UI_FONT, 9, "bold")).pack(side="left", padx=(8, 0))

    # ---- Card 2: Annotated Output ---------------------------------------
    def _build_output_card(self, parent):
        card = Card(parent, pad=18)
        card.grid(row=1, column=0, sticky="nsew")
        body = card.body
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        self._card_heading(body, "2", "Annotated Output",
                           "AI prediction result with highlighted region").grid(
            row=0, column=0, sticky="ew", pady=(0, 10))

        out_frame = tk.Frame(body, bg=IMAGE_CANVAS_BG, highlightthickness=1,
                             highlightbackground=BORDER_COLOR, bd=0)
        out_frame.grid(row=1, column=0, sticky="nsew")
        self.output_canvas = tk.Canvas(out_frame, bg=IMAGE_CANVAS_BG,
                                        highlightthickness=0, bd=0)
        self.output_canvas.pack(fill="both", expand=True)
        self.output_canvas.create_text(
            10, 10, text="Run an analysis to see the annotated result",
            fill="#8b93a5", font=(UI_FONT, 9), anchor="nw"
        )
        self.output_canvas.bind("<Configure>", lambda e: self._redraw_current_images())
        self.output_canvas.bind("<Button-1>", lambda e: self._on_click_output_image())

    # ---- Card 3: Findings & Report ---------------------------------------
    def _build_findings_card(self, parent):
        card = Card(parent, pad=18)
        card.grid(row=0, column=0, sticky="nsew")
        body = card.body
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)

        header_row = tk.Frame(body, bg=CARD_BG)
        header_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_row.columnconfigure(0, weight=1)

        self._card_heading(header_row, "3", "Findings & Report",
                           "Model prediction and detailed analysis").grid(
            row=0, column=0, sticky="w")

        self.download_button = PillButton(
            header_row, "Download Report", self.download_report,
            bg=BLUE_PRIMARY, fg="#ffffff", hover_bg=BLUE_PRIMARY_DARK,
            icon="\u2B07", width=200, height=38
        )
        self.download_button.grid(row=0, column=1, sticky="e")

        # Verdict banner
        self.verdict_canvas = tk.Canvas(body, height=92, bg=CARD_BG,
                                         highlightthickness=0, bd=0)
        self.verdict_canvas.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.verdict_icon_holder = tk.Frame(body, bg=CARD_BG)
        self._verdict_state = {"prediction": None, "model_name": "\u2014"}
        self.verdict_canvas.bind("<Configure>", lambda e: self._draw_verdict_banner())

        # Probability section
        prob_header = tk.Frame(body, bg=CARD_BG)
        prob_header.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        prob_icon_canvas = tk.Canvas(prob_header, width=28, height=28, bg=CARD_BG,
                                     highlightthickness=0, bd=0)
        prob_icon_canvas.pack(side="left")
        rounded_rect(prob_icon_canvas, 0, 0, 28, 28, 6, fill=BLUE_LIGHT_BG, outline="")
        prob_icon_canvas.create_text(14, 14, text="\U0001F4CA", font=("Segoe UI Symbol", 10),
                                     fill=BLUE_PRIMARY, anchor="center")

        tk.Label(prob_header, text="Prediction Probabilities", bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 11, "bold")).pack(side="left", padx=(12, 0))

        self.probability_frame = tk.Frame(body, bg=CARD_BG)
        self.probability_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._render_probability_placeholder()

        # Tabs
        tabs_wrap = tk.Frame(body, bg=CARD_BG)
        tabs_wrap.grid(row=4, column=0, sticky="nsew")
        tabs_wrap.columnconfigure(0, weight=1)
        tabs_wrap.rowconfigure(1, weight=1)

        self.tabs = PillTabs(
            tabs_wrap,
            ["Features vs Normal", "Detection Details", "Model Information"],
            self._on_tab_change
        )
        self.tabs.grid(row=0, column=0, sticky="ew")

        self.tab_pages_frame = tk.Frame(tabs_wrap, bg=CARD_BG)
        self.tab_pages_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.tab_pages_frame.columnconfigure(0, weight=1)
        self.tab_pages_frame.rowconfigure(0, weight=1)

        self.tab_features = tk.Frame(self.tab_pages_frame, bg=CARD_BG)
        self.tab_details = tk.Frame(self.tab_pages_frame, bg=CARD_BG)
        self.tab_model_info = tk.Frame(self.tab_pages_frame, bg=CARD_BG)
        for f in (self.tab_features, self.tab_details, self.tab_model_info):
            f.grid(row=0, column=0, sticky="nsew")

        self._build_features_tab(self.tab_features)
        self._build_details_tab(self.tab_details)
        self._build_model_info_tab(self.tab_model_info)

        self.tab_features.tkraise()

    def _card_heading(self, parent, number, title, subtitle):
        wrap = tk.Frame(parent, bg=CARD_BG)
        badge = tk.Canvas(wrap, width=32, height=32, bg=CARD_BG,
                          highlightthickness=0, bd=0)
        badge.pack(side="left")
        badge.create_oval(1, 1, 31, 31, fill=BLUE_PRIMARY, outline=BLUE_PRIMARY)
        badge.create_text(16, 16, text=number, fill="#ffffff", font=(UI_FONT, 11, "bold"))

        text_wrap = tk.Frame(wrap, bg=CARD_BG)
        text_wrap.pack(side="left", padx=(16, 0), anchor="w")
        tk.Label(text_wrap, text=title, bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 14, "bold"), anchor="w").pack(anchor="w")
        tk.Label(text_wrap, text=subtitle, bg=CARD_BG, fg=TEXT_MUTED,
                 font=(UI_FONT, 9), anchor="w").pack(anchor="w")
        return wrap

    def _render_probability_placeholder(self):
        for child in self.probability_frame.winfo_children():
            child.destroy()
        tk.Label(self.probability_frame, text="Run an analysis to see class probabilities.",
                 bg=CARD_BG, fg=TEXT_FAINT, font=(UI_FONT, 9, "italic")).pack(anchor="w")

    def _draw_verdict_banner(self):
        self.verdict_canvas.delete("all")
        w = self.verdict_canvas.winfo_width()
        h = max(self.verdict_canvas.winfo_height(), 82)
        if w < 10:
            return

        prediction = self._verdict_state["prediction"]
        model_name = self._verdict_state["model_name"]

        if prediction is None:
            bg, fg, icon_bg, glyph, title = NEUTRAL_BG, NEUTRAL_TEXT, TEXT_FAINT, "", "NO IMAGE SELECTED"
        else:
            style = VERDICT_STYLES.get(prediction, VERDICT_STYLES["Normal"])
            bg, fg, icon_bg, glyph, title = style["bg"], style["fg"], style["icon_bg"], style["icon"], prediction.upper()

        rounded_rect(self.verdict_canvas, 0, 0, w, h, 12, fill=bg, outline="")
        cy = h / 2

        # Circular icon badge: diameter 46px, centered vertically at cy
        self.verdict_canvas.create_oval(18, cy - 24, 66, cy + 24, fill=icon_bg, outline=icon_bg)
        self.verdict_canvas.create_text(42, cy, text=glyph, fill="#ffffff", font=(UI_FONT, 17, "bold"))

        # Verdict text column starts at x=84 (20px horizontal gap from icon badge)
        # Vertically spaced cleanly without overlapping
        self.verdict_canvas.create_text(88, cy - 27, text="VERDICT", fill=fg,
                                        font=(UI_FONT, 8, "bold"), anchor="w")
        self.verdict_canvas.create_text(88, cy, text=title, fill=fg,
                                        font=(UI_FONT, 17, "bold"), anchor="w")
        self.verdict_canvas.create_text(88, cy + 27, text=f"Model: {model_name}", fill=TEXT_MUTED,
                                        font=(UI_FONT, 9), anchor="w")

    # ---- Features tab -----------------------------------------------------
    def _build_features_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        columns = ("idx", "feature", "value", "normal", "difference", "status")
        self.feature_tree = ttk.Treeview(parent, columns=columns, show="headings", height=11)

        headings = {
            "idx": "#", "feature": "Feature", "value": "Image value",
            "normal": "Normal reference", "difference": "Difference", "status": "Comparison",
        }
        widths = {"idx": 32, "feature": 210, "value": 90, "normal": 130, "difference": 90, "status": 90}
        anchors = {"idx": "center", "feature": "w", "value": "center",
                   "normal": "center", "difference": "center", "status": "center"}

        for col in columns:
            self.feature_tree.heading(col, text=headings[col])
            self.feature_tree.column(col, width=widths[col], anchor=anchors[col])

        self.feature_tree.tag_configure("odd", background="#f7f9fd")
        self.feature_tree.tag_configure("even", background=CARD_BG)

        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.feature_tree.yview,
                               style="Clean.Vertical.TScrollbar")
        self.feature_tree.configure(yscrollcommand=scroll.set)

        self.feature_tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    # ---- Detection details tab --------------------------------------------
    def _build_details_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.details_text = tk.Text(parent, wrap="word", bg=CARD_BG, fg=TEXT_DARK,
                                    relief="flat", borderwidth=0, font=("Consolas", 9),
                                    padx=6, pady=6, height=12)
        self.details_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.details_text.yview,
                               style="Clean.Vertical.TScrollbar")
        self.details_text.configure(yscrollcommand=scroll.set, state="disabled")
        scroll.grid(row=0, column=1, sticky="ns")

    # ---- Model information tab --------------------------------------------
    def _build_model_info_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        self.model_info_frame = tk.Frame(parent, bg=CARD_BG)
        self.model_info_frame.grid(row=0, column=0, sticky="nsew")
        self._render_model_info()

    def _render_model_info(self):
        for child in self.model_info_frame.winfo_children():
            child.destroy()

        def row(label, value):
            r = tk.Frame(self.model_info_frame, bg=CARD_BG)
            r.pack(fill="x", pady=5)
            tk.Label(r, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                     font=(UI_FONT, 9), width=24, anchor="w").pack(side="left")
            tk.Label(r, text=value, bg=CARD_BG, fg=TEXT_DARK,
                     font=(UI_FONT, 9, "bold"), anchor="w", wraplength=340,
                     justify="left").pack(side="left", fill="x")

        row("Active model", self.model_name)
        row("Feature vector length", str(len(self.feature_names)))
        row("Classes", ", ".join(["Benign", "Malignant", "Normal"]))
        row("Model directory", self.model_dir)
        row("Reference dataset", self.features_path)

        if self.normal_stats:
            n = next(iter(self.normal_stats.values()))["n"]
            row("Normal reference samples", str(n))

        if self.training_metadata:
            md = self.training_metadata
            if "test_accuracy" in md:
                row("Held-out test accuracy", f"{md['test_accuracy'] * 100:.1f}%")
            if "calibrated_test_accuracy" in md:
                row("Calibrated test accuracy", f"{md['calibrated_test_accuracy'] * 100:.1f}%")
            if "n_samples" in md:
                row("Training samples used", str(md["n_samples"]))

        

    def _on_tab_change(self, index):
        pages = [self.tab_features, self.tab_details, self.tab_model_info]
        pages[index].tkraise()

    # ------------------------------------------------------------------
    # REPORTS PAGE
    # ------------------------------------------------------------------
    def _build_reports_page(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        card = Card(parent, pad=20)
        card.grid(row=0, column=0, sticky="nsew")
        body = card.body
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        tk.Label(body, text="Reports", bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 15, "bold"), anchor="w").grid(row=0, column=0, sticky="w",pady=(0, 6))
        tk.Label(body, text="Every analysis you run this session is listed here for review or re-download.",
                 bg=CARD_BG, fg=TEXT_MUTED, font=(UI_FONT, 9), anchor="w").grid(
            row=1, column=0, sticky="w", pady=(0, 18))

        list_wrap = tk.Frame(body, bg=CARD_BG)
        list_wrap.grid(row=2, column=0, sticky="nsew", pady=(46, 0))
        list_wrap.columnconfigure(0, weight=1)
        self.reports_list_frame = list_wrap
        self._render_reports_list()

    def _render_reports_list(self):
        for child in self.reports_list_frame.winfo_children():
            child.destroy()

        if not self.report_history:
            tk.Label(self.reports_list_frame,
                     text="No reports yet. Run an analysis on the Home tab, then "
                          "click \u201cDownload Report\u201d to save it \u2014 it will also "
                          "appear here.",
                     bg=CARD_BG, fg=TEXT_FAINT, font=(UI_FONT, 9, "italic"),
                     wraplength=520, justify="left").grid(row=0, column=0, sticky="w")
            return

        for i, record in enumerate(reversed(self.report_history)):
            self._render_report_row(self.reports_list_frame, i, record)

    def _render_report_row(self, parent, row_index, record):
        row_card = tk.Frame(parent, bg="#f7f9fd", highlightbackground=BORDER_COLOR,
                            highlightthickness=1, bd=0)
        row_card.grid(row=row_index, column=0, sticky="ew", pady=6)
        row_card.columnconfigure(1, weight=1)

        style = VERDICT_STYLES.get(record["prediction"], VERDICT_STYLES["Normal"])
        dot = tk.Canvas(row_card, width=38, height=38, bg="#f7f9fd", highlightthickness=0)
        dot.grid(row=0, column=0, padx=(16, 14), pady=10)
        dot.create_oval(2, 2, 36, 36, fill=style["icon_bg"], outline="")
        dot.create_text(19, 19, text=style["icon"], fill="#ffffff", font=(UI_FONT, 13, "bold"))

        info = tk.Frame(row_card, bg="#f7f9fd")
        info.grid(row=0, column=1, sticky="w", pady=10)
        tk.Label(info, text=f"{record['prediction']}  \u2014  {os.path.basename(record['image_path'])}",
                 bg="#f7f9fd", fg=TEXT_DARK, font=(UI_FONT, 10, "bold"), anchor="w").pack(anchor="w")
        tk.Label(info, text=f"Analyzed {record['timestamp']}  \u2022  Model: {record['model_name']}",
                 bg="#f7f9fd", fg=TEXT_MUTED, font=(UI_FONT, 8), anchor="w").pack(anchor="w", pady=(2, 0))

        btn_wrap = tk.Frame(row_card, bg="#f7f9fd")
        btn_wrap.grid(row=0, column=2, padx=16)
        PillButton(btn_wrap, "Download", lambda r=record: self._save_report(r),
                  bg="#ffffff", fg=BLUE_PRIMARY, hover_bg=BLUE_LIGHT_BG, border=BLUE_BORDER,
                  icon="\u2B07", width=128, height=32, font_size=9,
                  page_bg="#f7f9fd").pack()

    # ------------------------------------------------------------------
    # ABOUT PAGE
    # ------------------------------------------------------------------
    def _build_about_page(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        card = Card(parent, pad=24)
        card.grid(row=0, column=0, sticky="nsew")
        body = card.body
        body.columnconfigure(0, weight=1)

        tk.Label(body, text="About this tool", bg=CARD_BG, fg=TEXT_DARK,
                 font=(UI_FONT, 15, "bold"), anchor="w").pack(anchor="w")

        intro = (
            "Liver Ultrasound AI is a research prototype that screens liver "
            "ultrasound images and classifies them as Normal, Benign, or "
            "Malignant. It combines classic image processing with a trained "
            "machine-learning classifier."
        )
        tk.Label(body, text=intro, bg=CARD_BG, fg=TEXT_MUTED, font=(UI_FONT, 10),
                 wraplength=620, justify="left", anchor="w").pack(anchor="w", pady=(10, 18))

        steps = [
            ("1", "Preprocess", "Despeckling, denoising and CLAHE contrast enhancement of the "
                                "ultrasound frame."),
            ("2", "Segment", "The liver region of interest and any candidate lesion region are "
                             "located automatically."),
            ("3", "Extract features", "Intensity, texture (GLCM) and shape features are computed "
                                      "for the liver and any candidate mass."),
            ("4", "Classify", "A trained, calibrated classifier scores the feature vector and "
                              "outputs class probabilities."),
        ]
        for num, title, desc in steps:
            row = tk.Frame(body, bg=CARD_BG)
            row.pack(fill="x", pady=8, anchor="w")

            badge = tk.Canvas(row, width=26, height=26, bg=CARD_BG, highlightthickness=0, bd=0)
            badge.pack(side="left", anchor="n")
            badge.create_oval(1, 1, 25, 25, fill=BLUE_LIGHT_BG, outline=BLUE_BORDER, width=1)
            badge.create_text(13, 13, text=num, fill=BLUE_PRIMARY, font=(UI_FONT, 10, "bold"))

            content_col = tk.Frame(row, bg=CARD_BG)
            content_col.pack(side="left", padx=(14, 0), fill="x", expand=True)

            tk.Label(content_col, text=title, bg=CARD_BG, fg=TEXT_DARK,
                     font=(UI_FONT, 10, "bold"), anchor="w").pack(anchor="w")
            tk.Label(content_col, text=desc, bg=CARD_BG, fg=TEXT_MUTED, font=(UI_FONT, 9),
                     wraplength=520, justify="left", anchor="w").pack(anchor="w", pady=(2, 0))
        

    # ------------------------------------------------------------------
    # PAGE NAVIGATION
    # ------------------------------------------------------------------
    def show_page(self, name):
        pages = {"home": self.page_home, "reports": self.page_reports, "about": self.page_about}
        pages[name].tkraise()
        for key, btn in self.nav_buttons.items():
            btn.set_active(key == name)
        if name == "reports":
            self._render_reports_list()

    # ------------------------------------------------------------------
    # MODEL / REFERENCE DATA LOADING
    # ------------------------------------------------------------------
    def _load_model_and_reference(self):
        try:
            model_path = os.path.join(self.model_dir, "liver_classifier.joblib")
            scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")
            feature_path = os.path.join(self.model_dir, "feature_names.joblib")
            name_path = os.path.join(self.model_dir, "model_name.joblib")
            meta_path = os.path.join(self.model_dir, "training_metadata.joblib")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = list(joblib.load(feature_path))

            if os.path.exists(name_path):
                self.model_name = joblib.load(name_path)

            if os.path.exists(meta_path):
                try:
                    self.training_metadata = joblib.load(meta_path)
                except Exception:
                    self.training_metadata = None

            self._verdict_state["model_name"] = self.model_name
            self._draw_verdict_banner()

            self._load_normal_reference()
            self._render_model_info()

            self.status_var.set("Model loaded")

        except Exception as exc:
            self.status_var.set("Model load failed")
            messagebox.showerror("Model loading error", str(exc))

    def _load_normal_reference(self):
        if not os.path.exists(self.features_path):
            self.normal_stats = {}
            return

        data = np.load(self.features_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        stored_names = list(data["feature_names"])

        normal_mask = (y.astype(str) == "Normal")
        if not np.any(normal_mask):
            self.normal_stats = {}
            return

        normal_X = X[normal_mask]
        self.normal_stats = {}

        for i, name in enumerate(stored_names):
            if i >= normal_X.shape[1]:
                continue
            values = normal_X[:, i].astype(np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            self.normal_stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": int(values.size),
            }

    # ------------------------------------------------------------------
    # IMAGE SELECTION
    # ------------------------------------------------------------------
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select liver ultrasound image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.current_image_path = path

        try:
            with Image.open(path) as im:
                width, height = im.size
                fmt = (im.format or os.path.splitext(path)[1].lstrip(".").upper())
        except Exception:
            width = height = None
            fmt = os.path.splitext(path)[1].lstrip(".").upper()

        self.file_name_var.set(os.path.basename(path))
        if width and height:
            self.file_dims_var.set(f"{width} x {height} px")
        else:
            self.file_dims_var.set("Dimensions unavailable")
        self.file_format_var.set(f"{fmt} Image" if fmt else "")

        self._render_status_chip("Image loaded successfully", ok=True)

        self._verdict_state["prediction"] = None
        self._draw_verdict_banner()
        self.status_var.set("Image selected \u2014 ready to analyze")

        self._show_image_on_canvas(self.input_canvas, path, "input")

    # ------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------
    def analyze(self):
        if not self.current_image_path:
            messagebox.showinfo("Choose an image", "Please select an ultrasound image first.")
            return

        if self.model is None or self.scaler is None:
            messagebox.showerror("Model unavailable", "The trained model could not be loaded.")
            return

        self.analyze_button.set_enabled(False)
        self.choose_button.set_enabled(False)
        self.status_var.set("Analyzing...")

        image_path = self.current_image_path
        thread = threading.Thread(target=self._run_analysis_pipeline, args=(image_path,), daemon=True)
        thread.start()

    def _run_analysis_pipeline(self, image_path):
        try:
            pipeline_result = process_image_path(image_path)
            vector = pipeline_result["feature_vector"]

            expected_count = len(self.feature_names)
            if vector.shape[0] != expected_count:
                raise ValueError(
                    "Feature mismatch.\n\n"
                    f"Image pipeline produced {vector.shape[0]} features, but the "
                    f"model expects {expected_count}.\n\n"
                    "Make sure the feature dataset and model were rebuilt together "
                    "after any change to feature_extraction.py."
                )

            vector_2d = vector.reshape(1, -1)
            scaled = self.scaler.transform(vector_2d)

            clf = getattr(self.model, "estimator", self.model)
            prediction = str(clf.predict(scaled)[0])

            probabilities = {}
            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(scaled)[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)
                probabilities = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
                max_cls = max(probabilities, key=probabilities.get)
                if max_cls != prediction:
                    probabilities[prediction], probabilities[max_cls] = (
                        probabilities[max_cls],
                        probabilities[prediction],
                    )
            elif hasattr(clf, "predict_proba"):
                probabilities_array = clf.predict_proba(scaled)[0]
                probabilities = {str(c): float(p) for c, p in zip(clf.classes_, probabilities_array)}
            elif hasattr(self.model, "predict_proba"):
                probabilities_array = self.model.predict_proba(scaled)[0]
                probabilities = {str(c): float(p) for c, p in zip(self.model.classes_, probabilities_array)}

            result = {
                "image_path": image_path,
                "liver_bbox": pipeline_result["liver_bbox"],
                "mass_bbox": pipeline_result["mass_bbox"],
                "features": pipeline_result["features"],
                "prediction": prediction,
                "probabilities": probabilities,
            }

            annotated = self._annotate_result(pipeline_result["resized_bgr"], result)

            self.root.after(0, self._on_analysis_success, result, annotated, pipeline_result["resized_bgr"])

        except Exception as exc:
            self.root.after(0, self._on_analysis_error, exc)

    def _on_analysis_success(self, result, annotated, resized_bgr):
        self.current_result = result
        self.annotated_display = annotated

        self._update_verdict(result)
        self._update_probabilities(result)
        self._update_feature_table(result)
        self._update_details(result)

        self._redraw_current_images()

        self.status_var.set("Analysis complete")
        self.analyze_button.set_enabled(True)
        self.choose_button.set_enabled(True)

        self.report_history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "image_path": result["image_path"],
            "prediction": str(result["prediction"]),
            "probabilities": dict(result["probabilities"]),
            "features": dict(result["features"]),
            "liver_bbox": result["liver_bbox"],
            "mass_bbox": result["mass_bbox"],
            "model_name": self.model_name,
            "resized_bgr": resized_bgr.copy(),
            "annotated_bgr": annotated.copy(),
        })

    def _on_analysis_error(self, exc):
        self.status_var.set("Analysis failed")
        self.analyze_button.set_enabled(True)
        self.choose_button.set_enabled(True)
        messagebox.showerror("Analysis error", str(exc))

    def _annotate_result(self, resized_bgr, result):
        annotated = resized_bgr.copy()
        prediction = result["prediction"]
        color = LABEL_COLORS_BGR.get(prediction, (255, 255, 255))

        lx, ly, lw, lh = result["liver_bbox"]
        cv2.rectangle(annotated, (lx, ly), (lx + lw, ly + lh), (190, 190, 190), 1)

        confidence = result["probabilities"].get(prediction, 0.0)
        conf_str = f"  {confidence * 100:.1f}%" if confidence > 0 else ""

        if result["mass_bbox"] is not None and prediction != "Normal":
            mx, my, mw, mh = result["mass_bbox"]
            cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), color, 3)

            label = f"{prediction}{conf_str}"
            text_y = my - 10 if my - 10 > 20 else my + mh + 22
            cv2.putText(annotated, label, (mx, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.62, color, 2, cv2.LINE_AA)
        else:
            label = f"{prediction}{conf_str}"
            cv2.putText(annotated, label, (lx + 8, ly + 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.65, color, 2, cv2.LINE_AA)

        return annotated

    def _update_verdict(self, result):
        prediction = str(result["prediction"])
        self._verdict_state["prediction"] = prediction
        self._verdict_state["model_name"] = self.model_name
        self._draw_verdict_banner()

    def _update_probabilities(self, result):
        for child in self.probability_frame.winfo_children():
            child.destroy()

        probabilities = result["probabilities"]
        if not probabilities:
            tk.Label(self.probability_frame, text="Probability output unavailable.",
                     bg=CARD_BG, fg=TEXT_FAINT, font=(UI_FONT, 9, "italic")).pack(anchor="w")
            return

        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        for label, probability in ordered:
            color = BAR_COLORS.get(label, BLUE_PRIMARY)
            bar = ProbabilityBar(self.probability_frame, label, float(probability), color)
            bar.pack(fill="x", pady=4)

    def _format_value(self, name, value):
        if name == "mass_detected":
            return "Yes" if value >= 0.5 else "No"
        if name in PERCENT_FEATURES:
            return f"{value * 100:.2f}%"
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.4f}"

    def _update_feature_table(self, result):
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        features = result["features"]

        for idx, name in enumerate(self.feature_names, start=1):
            if name not in features:
                continue

            value = float(features[name])
            label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
            normal = self.normal_stats.get(name)

            if normal is None:
                normal_text = "N/A"
                difference_text = "N/A"
                status = "No reference"
            else:
                mean = normal["mean"]
                std = normal["std"]
                normal_text = f"{self._format_value(name, mean)} \u00b1 {self._format_value(name, std)}"
                difference = value - mean
                difference_text = f"{difference:+.4f}"

                if name == "mass_detected":
                    status = "Above normal" if (value >= 0.5 and mean < 0.5) else "Near normal"
                elif std < 1e-9:
                    status = "Near normal"
                else:
                    z = abs(difference / std)
                    if z <= 1:
                        status = "Within 1 SD"
                    elif z <= 2:
                        status = "1\u20132 SD"
                    else:
                        status = ">2 SD"

            tag = "odd" if idx % 2 else "even"
            self.feature_tree.insert(
                "", "end", tags=(tag,),
                values=(idx, label, self._format_value(name, value), normal_text, difference_text, status)
            )

    def _update_details(self, result):
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")

        liver_bbox = result["liver_bbox"]
        mass_bbox = result["mass_bbox"]
        probabilities = result["probabilities"]
        features = result["features"]

        lines = []
        lines.append("IMAGE\n")
        lines.append(f"{result['image_path']}\n\n")

        lines.append("DETECTION\n")
        lines.append(f"Liver ROI: {liver_bbox}\n")

        if mass_bbox is None:
            lines.append("Candidate region: none detected\n")
        else:
            lines.append(f"Candidate bounding box: {mass_bbox}\n")

        detected = features.get("mass_detected", 0)
        lines.append(f"Candidate detected: {'Yes' if detected else 'No'}\n\n")

        lines.append("MODEL OUTPUT\n")
        if probabilities:
            for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"{label}: {probability * 100:.2f}%\n")

        lines.append("\n")
        lines.append("NORMAL REFERENCE\n")
        lines.append(
            "The 'Normal reference' values are calculated from the Normal samples "
            "in the project's cached training feature dataset. They are NOT clinical "
            "reference ranges.\n\n"
        )

        if self.normal_stats:
            n = next(iter(self.normal_stats.values()))["n"]
            lines.append(f"Normal samples used for reference: {n}\n\n")
        else:
            lines.append("Normal reference dataset unavailable.\n\n")

        lines.append("INTERPRETATION NOTE\n")
        lines.append(
            "Values outside the Normal dataset mean \u00b1 standard deviation are "
            "highlighted for comparison. This is a statistical comparison against "
            "this project's training data, not a medical diagnostic threshold.\n"
        )

        self.details_text.insert("1.0", "".join(lines))
        self.details_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # IMAGE RENDERING
    # ------------------------------------------------------------------
    def _redraw_current_images(self):
        if self.current_image_path and os.path.exists(self.current_image_path):
            self._show_image_on_canvas(self.input_canvas, self.current_image_path, "input")

        if self.annotated_display is not None:
            self._show_cv_image_on_canvas(self.output_canvas, self.annotated_display, "output",
                                          label=self._verdict_state.get("prediction"))

    def _show_image_on_canvas(self, canvas, path, tag):
        try:
            image = Image.open(path).convert("RGB")
            self.original_pil = image
            self._draw_pil_image(canvas, image, tag)
            canvas.configure(cursor="hand2")
        except Exception:
            canvas.delete("all")

    def _show_cv_image_on_canvas(self, canvas, image_bgr, tag, label=None):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._draw_pil_image(canvas, image, tag, chip_label=(f"Predicted: {label}" if label else None))
        canvas.configure(cursor="hand2")

    def _draw_pil_image(self, canvas, image, tag, chip_label=None):
        canvas.delete("all")

        width = max(canvas.winfo_width(), 100)
        height = max(canvas.winfo_height(), 100)

        image = image.copy()
        image.thumbnail((width - 16, height - 16), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        canvas.create_image(width // 2, height // 2, image=photo, anchor="center")

        if tag == "input":
            self.original_display = photo
        else:
            self.annotated_display_photo = photo

        if chip_label:
            img_left = width // 2 - image.width // 2
            img_top = height // 2 - image.height // 2
            chip_style = VERDICT_STYLES.get(self._verdict_state.get("prediction"), VERDICT_STYLES["Normal"])
            pad_x, pad_y = 10, 6
            text_w = 8 * len(chip_label)
            x0, y0 = img_left + 10, img_top + 10
            x1, y1 = x0 + text_w + pad_x * 2, y0 + 22
            rounded_rect(canvas, x0, y0, x1, y1, 6, fill=chip_style["icon_bg"], outline="")
            canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=chip_label,
                               fill="#ffffff", font=(UI_FONT, 9, "bold"))

    # ------------------------------------------------------------------
    # ENLARGED IMAGE VIEWER ACTIONS
    # ------------------------------------------------------------------
    def open_image_viewer(self, image, title="Ultrasound Inspection Viewer"):
        """Open an enlarged, zoomable, draggable viewer window for a PIL Image."""
        if image is None:
            return None
        return open_image_viewer(image, title, parent=self.root)

    def _on_click_input_image(self):
        if not self.current_image_path:
            return
        try:
            if self.original_pil is not None:
                image = self.original_pil
            elif os.path.exists(self.current_image_path):
                image = Image.open(self.current_image_path).convert("RGB")
                self.original_pil = image
            else:
                return
            filename = os.path.basename(self.current_image_path)
            self.open_image_viewer(image, f"Input Ultrasound Image \u2014 {filename}")
        except Exception as exc:
            messagebox.showerror("Viewer error", f"Could not open image viewer: {exc}")

    def _on_click_output_image(self):
        if self.annotated_display is None:
            return
        try:
            rgb = cv2.cvtColor(self.annotated_display, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            filename = os.path.basename(self.current_image_path) if self.current_image_path else "Image"
            pred = self._verdict_state.get("prediction")
            suffix = f" (Verdict: {pred})" if pred else ""
            self.open_image_viewer(image, f"Annotated Ultrasound Analysis \u2014 {filename}{suffix}")
        except Exception as exc:
            messagebox.showerror("Viewer error", f"Could not open image viewer: {exc}")

    # ------------------------------------------------------------------
    # REPORT EXPORT
    # ------------------------------------------------------------------
    def download_report(self):
        if self.current_result is None:
            messagebox.showinfo("No analysis yet", "Run an analysis before downloading a report.")
            return

        record = self.report_history[-1] if self.report_history else None
        if record is None:
            messagebox.showinfo("No analysis yet", "Run an analysis before downloading a report.")
            return

        self._save_report(record)

    def _save_report(self, record):
        default_name = os.path.splitext(os.path.basename(record["image_path"]))[0] + "_report.pdf"
        path = filedialog.asksaveasfilename(
            title="Save analysis report",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF Document (*.pdf)", "*.pdf"), ("HTML report (*.html)", "*.html"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            if path.lower().endswith(".html"):
                html = self._build_report_html(record)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
            else:
                self._build_report_pdf(record, path)

            self.status_var.set(f"Report saved to {os.path.basename(path)}")
            messagebox.showinfo("Report saved", f"The analysis report was successfully saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Could not save report", str(exc))

    def _save_report_html(self, record):
        default_name = os.path.splitext(os.path.basename(record["image_path"]))[0] + "_report.html"
        path = filedialog.asksaveasfilename(
            title="Save analysis report",
            defaultextension=".html",
            initialfile=default_name,
            filetypes=[("HTML report", "*.html"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            html = self._build_report_html(record)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            self.status_var.set(f"Report saved to {os.path.basename(path)}")
            messagebox.showinfo("Report saved", f"The analysis report was saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Could not save report", str(exc))

    def _build_report_pdf(self, record, filepath):
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor(TEXT_DARK),
        )
        subtitle_style = ParagraphStyle(
            "DocSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor(TEXT_MUTED),
        )
        normal_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor(TEXT_DARK),
        )

        story = []

        prediction = record["prediction"]
        pred_colors = {
            "Normal": (colors.HexColor("#186b39"), colors.HexColor("#e5f6ea"), colors.HexColor("#1f9d55")),
            "Benign": (colors.HexColor("#8a5a06"), colors.HexColor("#fdf3e0"), colors.HexColor("#c9820a")),
            "Malignant": (colors.HexColor("#a3221c"), colors.HexColor("#fdecec"), colors.HexColor("#dc2626")),
        }
        fg, bg, border_col = pred_colors.get(
            prediction,
            (colors.HexColor("#152238"), colors.HexColor("#eef2f9"), colors.HexColor("#c7d2e6")),
        )

        # Header Title
        story.append(Paragraph(f"<b>{APP_TITLE}</b>", title_style))
        story.append(Spacer(1, 2))
        story.append(Paragraph("Automated Quantitative Ultrasound Feature Analysis &amp; Screening Report", subtitle_style))
        story.append(Spacer(1, 3))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(BLUE_PRIMARY), spaceAfter=8))

        # Metadata table
        meta_data = [
            [
                Paragraph(f"<b>Scan File:</b> {os.path.basename(record['image_path'])}", normal_style),
                Paragraph(f"<b>Analyzed Date:</b> {record['timestamp']}", normal_style),
            ],
            [
                Paragraph(f"<b>Model Engine:</b> {record['model_name']}", normal_style),
                Paragraph("<b>Clinical Status:</b> Automated Screening Complete", normal_style),
            ],
        ]
        t_meta = Table(meta_data, colWidths=[270, 270])
        t_meta.setStyle(
            TableStyle([
                ("PADDING", (0, 0), (-1, -1), 3),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fd")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER_COLOR)),
            ])
        )
        story.append(t_meta)
        story.append(Spacer(1, 7))

        # Verdict banner
        confidence = record["probabilities"].get(prediction, 0.0)
        mass_detected_str = "Detected" if record.get("mass_bbox") and prediction != "Normal" else "No focal lesion detected"
        v_table = Table(
            [[
                Paragraph(
                    f"<b>VERDICT: {prediction.upper()}</b>",
                    ParagraphStyle("VB", fontName="Helvetica-Bold", fontSize=11, textColor=fg),
                ),
                Paragraph(
                    f"Confidence: <b>{confidence * 100:.1f}%</b> &nbsp;|&nbsp; Region: <b>{mass_detected_str}</b>",
                    normal_style,
                ),
            ]],
            colWidths=[270, 270],
        )
        v_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 1, border_col),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        story.append(v_table)
        story.append(Spacer(1, 7))

        # Class Probabilities table
        prob_rows = [
            [
                Paragraph("<b>Class</b>", normal_style),
                Paragraph("<b>Probability</b>", normal_style),
                Paragraph("<b>Clinical Interpretation</b>", normal_style),
            ]
        ]
        ordered_probs = sorted(record["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        interpretations = {
            "Normal": "Healthy parenchymal echotexture, non-fatty liver",
            "Benign": "Diffuse changes / fatty liver / benign lesion",
            "Malignant": "Suspicious focal hypoechoic lesion / potential malignancy",
        }
        for cls_name, p in ordered_probs:
            prob_rows.append([
                cls_name,
                f"{p * 100:.1f}%",
                interpretations.get(cls_name, ""),
            ])
        t_prob = Table(prob_rows, colWidths=[90, 80, 370])
        t_prob.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE_LIGHT_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(BLUE_PRIMARY_DARK)),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER_COLOR)),
                ("PADDING", (0, 0), (-1, -1), 3),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(t_prob)
        story.append(Spacer(1, 7))

        # Side-by-side Images
        success1, buf1 = cv2.imencode(".png", record["resized_bgr"])
        success2, buf2 = cv2.imencode(".png", record["annotated_bgr"])
        if success1 and success2:
            im1 = RLImage(io.BytesIO(buf1.tobytes()), width=255, height=165)
            im2 = RLImage(io.BytesIO(buf2.tobytes()), width=255, height=165)
            img_table = Table(
                [[im1, im2],
                 [Paragraph("<font size=7 color='#6b7688'>Preprocessed Input Scan</font>", normal_style),
                  Paragraph("<font size=7 color='#6b7688'>Annotated Detection Output</font>", normal_style)]],
                colWidths=[270, 270],
            )
            img_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 1),
                ])
            )
            story.append(img_table)
            story.append(Spacer(1, 7))

        # Page 2: Detailed Quantitative Feature Analysis
        story.append(PageBreak())

        story.append(Paragraph("<b>Quantitative Ultrasound Feature Analysis</b>", title_style))
        story.append(Spacer(1, 2))
        story.append(Paragraph("Comparison of 20 IVP features against Normal liver reference baseline values", subtitle_style))
        story.append(Spacer(1, 3))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(BLUE_PRIMARY), spaceAfter=8))

        # Quantitative Features Table
        feat_rows = [
            [
                Paragraph("<b>#</b>", normal_style),
                Paragraph("<b>Extracted Quantitative Feature</b>", normal_style),
                Paragraph("<b>Scan Value</b>", normal_style),
                Paragraph("<b>Normal Reference</b>", normal_style),
                Paragraph("<b>Deviation</b>", normal_style),
            ]
        ]
        for idx, name in enumerate(self.feature_names, start=1):
            if name not in record["features"]:
                continue
            val = float(record["features"][name])
            label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
            normal = self.normal_stats.get(name)
            if normal is None:
                norm_text, diff_text = "N/A", "N/A"
            else:
                norm_text = f"{self._format_value(name, normal['mean'])} \u00b1 {self._format_value(name, normal['std'])}"
                diff = val - normal["mean"]
                diff_text = f"{diff:+.4f}"
            feat_rows.append([
                str(idx),
                label,
                self._format_value(name, val),
                norm_text,
                diff_text,
            ])

        t_feat = Table(feat_rows, colWidths=[20, 200, 100, 130, 90])
        t_feat.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE_LIGHT_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(BLUE_PRIMARY_DARK)),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER_COLOR)),
                ("PADDING", (0, 0), (-1, -1), 3),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(t_feat)
        story.append(Spacer(1, 10))

        # Disclaimer
        disc = Table(
            [[
                Paragraph(
                    "<font size=7 color='#8a5a06'><b>CLINICAL DISCLAIMER:</b> This automated quantitative report is "
                    "generated for decision support and research evaluation purposes only. It is not an automated medical "
                    "diagnosis. Final diagnostic evaluation must always be performed by a certified medical specialist.</font>",
                    normal_style,
                )
            ]],
            colWidths=[540],
        )
        disc.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(AMBER_BG)),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(AMBER)),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(disc)

        doc.build(story)

    @staticmethod
    def _bgr_to_base64_png(image_bgr):
        success, buffer = cv2.imencode(".png", image_bgr)
        if not success:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def _build_report_html(self, record):
        prediction = record["prediction"]
        style = VERDICT_STYLES.get(prediction, VERDICT_STYLES["Normal"])
        probabilities = sorted(record["probabilities"].items(), key=lambda kv: kv[1], reverse=True)

        input_b64 = self._bgr_to_base64_png(record["resized_bgr"])
        annotated_b64 = self._bgr_to_base64_png(record["annotated_bgr"])

        prob_rows = "".join(
            f"<tr><td>{cls}</td><td>{p * 100:.2f}%</td></tr>"
            for cls, p in probabilities
        )

        feature_rows = []
        for idx, name in enumerate(self.feature_names, start=1):
            if name not in record["features"]:
                continue
            value = float(record["features"][name])
            label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
            normal = self.normal_stats.get(name)
            if normal is None:
                normal_text, diff_text = "N/A", "N/A"
            else:
                normal_text = f"{self._format_value(name, normal['mean'])} \u00b1 {self._format_value(name, normal['std'])}"
                diff_text = f"{value - normal['mean']:+.4f}"
            feature_rows.append(
                f"<tr><td>{idx}</td><td>{label}</td><td>{self._format_value(name, value)}</td>"
                f"<td>{normal_text}</td><td>{diff_text}</td></tr>"
            )
        feature_rows_html = "".join(feature_rows)

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Liver Ultrasound AI \u2014 Analysis Report</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; background:{PAGE_BG}; color:{TEXT_DARK}; margin:0; padding:30px; }}
  .card {{ background:{CARD_BG}; border:1px solid {BORDER_COLOR}; border-radius:14px; padding:24px; margin-bottom:20px; }}
  h1 {{ margin:0 0 4px 0; font-size:22px; }}
  .subtitle {{ color:{TEXT_MUTED}; font-size:13px; margin-bottom:0; }}
  .verdict {{ background:{style['bg']}; color:{style['fg']}; border-radius:12px; padding:18px 22px; display:flex; align-items:center; gap:16px; }}
  .verdict .label {{ font-size:11px; letter-spacing:1px; font-weight:bold; opacity:0.8; }}
  .verdict .value {{ font-size:24px; font-weight:bold; }}
  .verdict .model {{ font-size:12px; color:{TEXT_MUTED}; }}
  table {{ border-collapse: collapse; width:100%; font-size:13px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid {BORDER_COLOR}; }}
  th {{ background:{BLUE_LIGHT_BG}; color:{TEXT_DARK}; }}
  img {{ max-width:100%; border-radius:10px; border:1px solid {BORDER_COLOR}; }}
  .imgs {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .imgs div {{ flex:1; min-width:260px; }}
  .imgs p {{ font-size:12px; color:{TEXT_MUTED}; margin:6px 0 0 0; }}
  .disclaimer {{ background:{AMBER_BG}; color:{AMBER_TEXT}; border:1px solid {AMBER}; border-radius:10px; padding:14px; font-size:12px; }}
  .muted {{ color:{TEXT_MUTED}; font-size:12px; }}
</style>
</head>
<body>

  <div class="card">
    <h1>{APP_TITLE}</h1>
    <p class="subtitle">{APP_SUBTITLE}</p>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Findings &amp; Report</h2>
    <div class="verdict">
      <div>
        <div class="label">VERDICT</div>
        <div class="value">{prediction.upper()}</div>
        <div class="model">Model: {record['model_name']}</div>
      </div>
    </div>
    <h3>Prediction probabilities</h3>
    <table>
      <tr><th>Class</th><th>Probability</th></tr>
      {prob_rows}
    </table>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Images</h2>
    <div class="imgs">
      <div>
        <img src="data:image/png;base64,{input_b64}" alt="Input image">
        <p>Input image \u2014 {os.path.basename(record['image_path'])}</p>
      </div>
      <div>
        <img src="data:image/png;base64,{annotated_b64}" alt="Annotated output">
        <p>Annotated output with detected liver and candidate region</p>
      </div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Features vs Normal reference</h2>
    <table>
      <tr><th>#</th><th>Feature</th><th>Image value</th><th>Normal reference</th><th>Difference</th></tr>
      {feature_rows_html}
    </table>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Detection details</h2>
    <p class="muted">Liver ROI: {record['liver_bbox']}</p>
    <p class="muted">Candidate bounding box: {record['mass_bbox'] if record['mass_bbox'] else 'no candidate region detected'}</p>
    <p class="muted">Generated: {record['timestamp']}</p>
  </div>

  <div class="disclaimer">
    This is an automated screening output, not a medical diagnosis. Final
    interpretation must be made by a qualified medical professional.
  </div>

</body>
</html>
"""
        return html


def main():
    parser = argparse.ArgumentParser(description="Single-window liver ultrasound analysis UI.")
    parser.add_argument("--model-dir", default="models", help="Trained model folder. Default: models")
    parser.add_argument("--features", default="features_dataset.npz",
                        help="Feature dataset used for Normal reference values.")
    args = parser.parse_args()

    root = tk.Tk()
    LiverClassifierUI(root, model_dir=args.model_dir, features_path=args.features)
    root.mainloop()


if __name__ == "__main__":
    main()