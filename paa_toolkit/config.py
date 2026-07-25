"""
Configuration module for the PAA Publication Toolkit.

This module centralizes all project paths and publication settings.
Nothing outside this module should hardcode directories.
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class SpringerStyle:
    """Journal-specific plotting configuration."""

    # Figure sizes (inches)
    SINGLE_COLUMN_WIDTH = 3.31      # 84 mm
    DOUBLE_COLUMN_WIDTH = 6.85      # 174 mm

    # Export
    DPI = 600

    # Typography
    FONT_FAMILY = "Arial"
    FONT_SIZE = 9
    TITLE_SIZE = 10
    LEGEND_SIZE = 8

    # Output formats
    EXPORT_PDF = True
    EXPORT_SVG = True
    EXPORT_PNG = True

    # Grid
    GRID_ALPHA = 0.25


class ProjectConfig:
    """
    Stores all project paths.
    """

    def __init__(self, root: str | Path):

        self.root = Path(root)

        self.input = self.root / "input"

        self.output = self.root / "output"

        self.figures = self.output / "figures"

        self.tables = self.output / "tables"

        self.reports = self.output / "reports"

        self.templates = self.root / "templates"

    def create_directories(self):

        self.output.mkdir(exist_ok=True)

        self.figures.mkdir(exist_ok=True)

        self.tables.mkdir(exist_ok=True)

        self.reports.mkdir(exist_ok=True)

    def summary(self):

        return {
            "Project": self.root,
            "Input": self.input,
            "Output": self.output,
            "Figures": self.figures,
            "Tables": self.tables,
            "Reports": self.reports,
            "Templates": self.templates
        }