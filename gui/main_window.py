"""
PyQt5 GUI for Fingerprint Matcher v3.

Workflow:
    1.  Select probe folder   (latent fingerprints).
    2.  Select gallery folder  (clean fingerprints).
    3.  Click "Run Matching".
    4.  For each probe image the pipeline runs against all gallery images.
    5.  Results are displayed in a ranked table with scores.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QComboBox,
    QGroupBox,
    QSplitter,
    QTextEdit,
    QSpinBox,
    QMessageBox,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont

from datetime import datetime

import cv2
import numpy as np

from ..pipeline import MatchingPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker thread — runs matching in the background
# ---------------------------------------------------------------------------

class MatchWorker(QThread):
    """Runs the matching pipeline on a background thread."""

    progress = pyqtSignal(int, int)          # (current, total)
    result_ready = pyqtSignal(str, list)     # (probe_name, results)
    error = pyqtSignal(str, str)             # (probe_name, error_msg)
    all_done = pyqtSignal(float)             # total_seconds

    def __init__(
        self,
        probe_paths: List[str],
        gallery_dir: str,
        n_workers: int = 4,
        log_path: Optional[str] = None,
    ):
        super().__init__()
        self.probe_paths = probe_paths
        self.gallery_dir = gallery_dir
        self.n_workers = n_workers
        self.log_path = log_path
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        t0 = time.perf_counter()
        pipeline = MatchingPipeline(
            n_workers=self.n_workers,
            log_path=self.log_path,
        )
        for probe_path in self.probe_paths:
            if self._stop_flag:
                break
            probe_name = Path(probe_path).stem
            try:
                results = pipeline.run(
                    probe_path=probe_path,
                    gallery_dir=self.gallery_dir,
                    progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
                )
                self.result_ready.emit(probe_name, results)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Error matching {probe_name}: {tb}")
                self.error.emit(probe_name, str(e))

        elapsed = time.perf_counter() - t0
        self.all_done.emit(elapsed)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fingerprint Matcher v2  —  SIFT + Minutiae + Texture")
        self.setMinimumSize(1100, 700)

        self._probe_dir: Optional[str] = None
        self._probe_single: Optional[str] = None   # single image path
        self._gallery_dir: Optional[str] = None
        self._worker: Optional[MatchWorker] = None
        self._all_results: Dict[str, List[Dict[str, Any]]] = {}

        self._build_ui()
        self.statusBar().showMessage("Ready")

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ---- Folder selection ----
        folder_group = QGroupBox("Input")
        fg_layout = QVBoxLayout(folder_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Probe (latent):"))
        self._lbl_probe = QLabel("(not selected)")
        self._lbl_probe.setStyleSheet("color: #666;")
        row1.addWidget(self._lbl_probe, 1)
        btn_probe_img = QPushButton("Single Image…")
        btn_probe_img.setToolTip("Select one fingerprint image to identify")
        btn_probe_img.clicked.connect(self._select_probe_image)
        row1.addWidget(btn_probe_img)
        btn_probe_dir = QPushButton("Folder…")
        btn_probe_dir.setToolTip("Select a folder of probe images to match in batch")
        btn_probe_dir.clicked.connect(self._select_probe_dir)
        row1.addWidget(btn_probe_dir)
        fg_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Gallery folder (clean):"))
        self._lbl_gallery = QLabel("(not selected)")
        self._lbl_gallery.setStyleSheet("color: #666;")
        row2.addWidget(self._lbl_gallery, 1)
        btn_gallery = QPushButton("Browse…")
        btn_gallery.clicked.connect(self._select_gallery_dir)
        row2.addWidget(btn_gallery)
        fg_layout.addLayout(row2)

        root_layout.addWidget(folder_group)

        # ---- Controls ----
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Workers:"))
        self._spin_workers = QSpinBox()
        self._spin_workers.setRange(1, 16)
        self._spin_workers.setValue(4)
        ctrl_layout.addWidget(self._spin_workers)

        ctrl_layout.addStretch()

        self._btn_run = QPushButton("▶  Run Matching")
        self._btn_run.setFixedHeight(36)
        self._btn_run.setStyleSheet(
            "QPushButton { background: #2a7de1; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 0 24px; }"
            "QPushButton:hover { background: #1b6fd0; }"
            "QPushButton:disabled { background: #999; }"
        )
        self._btn_run.clicked.connect(self._run_matching)
        ctrl_layout.addWidget(self._btn_run)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedHeight(36)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_matching)
        ctrl_layout.addWidget(self._btn_stop)

        self._btn_report = QPushButton("📄  Generate Report")
        self._btn_report.setFixedHeight(36)
        self._btn_report.setEnabled(False)
        self._btn_report.setStyleSheet(
            "QPushButton { background: #28a745; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 0 24px; }"
            "QPushButton:hover { background: #218838; }"
            "QPushButton:disabled { background: #999; }"
        )
        self._btn_report.clicked.connect(self._generate_report)
        ctrl_layout.addWidget(self._btn_report)

        root_layout.addLayout(ctrl_layout)

        # ---- Progress ----
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setValue(0)
        root_layout.addWidget(self._progress)

        # ---- Results area (splitter: table | image preview) ----
        splitter = QSplitter(Qt.Horizontal)

        # Left: probe selector + results table
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        probe_row = QHBoxLayout()
        probe_row.addWidget(QLabel("Probe:"))
        self._combo_probe = QComboBox()
        self._combo_probe.currentTextChanged.connect(self._on_probe_changed)
        probe_row.addWidget(self._combo_probe, 1)
        left_layout.addLayout(probe_row)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Rank", "Gallery ID", "Fused", "SIFT", "Minutiae", "Texture"]
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in [0, 2, 3, 4, 5]:
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.currentCellChanged.connect(self._on_row_changed)
        left_layout.addWidget(self._table)

        splitter.addWidget(left)

        # Right: image previews
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Probe image:"))
        self._lbl_probe_img = QLabel()
        self._lbl_probe_img.setAlignment(Qt.AlignCenter)
        self._lbl_probe_img.setMinimumSize(200, 200)
        self._lbl_probe_img.setStyleSheet("background: #f0f0f0; border: 1px solid #ccc;")
        right_layout.addWidget(self._lbl_probe_img)

        right_layout.addWidget(QLabel("Gallery match:"))
        self._lbl_gallery_img = QLabel()
        self._lbl_gallery_img.setAlignment(Qt.AlignCenter)
        self._lbl_gallery_img.setMinimumSize(200, 200)
        self._lbl_gallery_img.setStyleSheet("background: #f0f0f0; border: 1px solid #ccc;")
        right_layout.addWidget(self._lbl_gallery_img)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root_layout.addWidget(splitter, 1)

    # ---- Folder / image selection ------------------------------------------

    def _select_probe_image(self):
        """Select a single probe fingerprint image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Probe Fingerprint Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._probe_single = path
            self._probe_dir = None  # clear folder selection
            name = Path(path).name
            self._lbl_probe.setText(f"[Image] {name}")
            self._lbl_probe.setStyleSheet("color: #000;")

    def _select_probe_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Probe Folder")
        if d:
            self._probe_dir = d
            self._probe_single = None  # clear single-image selection
            self._lbl_probe.setText(f"[Folder] {d}")
            self._lbl_probe.setStyleSheet("color: #000;")

    def _select_gallery_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Gallery Folder")
        if d:
            self._gallery_dir = d
            self._lbl_gallery.setText(d)
            self._lbl_gallery.setStyleSheet("color: #000;")

    # ---- Matching ----------------------------------------------------------

    def _run_matching(self):
        if not self._gallery_dir:
            QMessageBox.warning(
                self, "Missing Gallery",
                "Please select the gallery (clean fingerprints) folder first."
            )
            return

        if not self._probe_single and not self._probe_dir:
            QMessageBox.warning(
                self, "Missing Probe",
                "Please select a probe image or probe folder first."
            )
            return

        # Build the list of probe image paths
        if self._probe_single:
            # Single-image mode
            probe_paths = [self._probe_single]
            log_dir = str(Path(self._probe_single).parent)
        else:
            # Folder mode — discover all images
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            probe_paths = sorted(
                str(p)
                for p in Path(self._probe_dir).iterdir()
                if p.suffix.lower() in exts and p.is_file()
            )
            log_dir = self._probe_dir

            if not probe_paths:
                QMessageBox.warning(
                    self, "No Images",
                    f"No images found in:\n{self._probe_dir}"
                )
                return

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setValue(0)
        self._all_results.clear()
        self._combo_probe.clear()
        self._table.setRowCount(0)

        # Log file next to the probe
        log_path = os.path.join(log_dir, "matching_log.txt")
        log_path = os.path.abspath(log_path)

        self._worker = MatchWorker(
            probe_paths=probe_paths,
            gallery_dir=self._gallery_dir,
            n_workers=self._spin_workers.value(),
            log_path=log_path,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

        self.statusBar().showMessage(f"Matching {len(probe_paths)} probes…")

    def _stop_matching(self):
        if self._worker:
            self._worker.stop()
            self.statusBar().showMessage("Stopping…")

    # ---- Callbacks ---------------------------------------------------------

    def _on_progress(self, current: int, total: int):
        pct = int(100 * current / max(total, 1))
        self._progress.setValue(pct)
        self._progress.setFormat(f"{current}/{total}  ({pct}%)")

    def _on_result(self, probe_name: str, results: list):
        self._all_results[probe_name] = results
        self._combo_probe.addItem(probe_name)
        if self._combo_probe.count() == 1:
            self._combo_probe.setCurrentIndex(0)
            self._populate_table(probe_name)
        self.statusBar().showMessage(
            f"Done: {probe_name}  —  top match: "
            f"{results[0]['gallery_id']} ({results[0]['fused_score']:.4f})"
            if results else f"Done: {probe_name}"
        )

    def _on_error(self, probe_name: str, msg: str):
        self.statusBar().showMessage(f"Error on {probe_name}: {msg}")
        logger.error(f"Error on {probe_name}: {msg}")

    def _on_all_done(self, elapsed: float):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_report.setEnabled(True)
        self._progress.setValue(100)
        self._elapsed = elapsed
        self.statusBar().showMessage(
            f"All done in {elapsed:.1f}s  —  "
            f"{len(self._all_results)} probes processed"
        )

    # ---- Table population --------------------------------------------------

    def _on_probe_changed(self, probe_name: str):
        if probe_name:
            self._populate_table(probe_name)
            self._show_probe_image(probe_name)

    def _populate_table(self, probe_name: str):
        results = self._all_results.get(probe_name, [])
        self._table.setRowCount(len(results))

        for i, r in enumerate(results):
            self._table.setItem(i, 0, self._num_item(i + 1))
            self._table.setItem(i, 1, QTableWidgetItem(r["gallery_id"]))
            self._table.setItem(i, 2, self._score_item(r.get("fused_score", 0)))
            self._table.setItem(i, 3, self._score_item(r.get("sift_score", 0)))
            self._table.setItem(i, 4, self._score_item(r.get("minutiae_score", 0)))
            self._table.setItem(i, 5, self._score_item(r.get("texture_score", 0)))

        if results:
            self._table.selectRow(0)

    def _on_row_changed(self, row, col, prev_row, prev_col):
        if row < 0:
            return
        probe_name = self._combo_probe.currentText()
        results = self._all_results.get(probe_name, [])
        if row < len(results):
            gallery_path = results[row].get("gallery_path", "")
            self._show_gallery_image(gallery_path)

    # ---- Image display -----------------------------------------------------

    def _show_probe_image(self, probe_name: str):
        # Single-image mode — we already know the exact path
        if self._probe_single:
            if Path(self._probe_single).stem == probe_name:
                self._display_image(self._probe_single, self._lbl_probe_img)
                return

        # Folder mode — search for the file by stem
        search_dir = self._probe_dir
        if not search_dir:
            if self._probe_single:
                search_dir = str(Path(self._probe_single).parent)
            else:
                return

        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            path = os.path.join(search_dir, probe_name + ext)
            if os.path.isfile(path):
                self._display_image(path, self._lbl_probe_img)
                return

    def _show_gallery_image(self, path: str):
        if path and os.path.isfile(path):
            self._display_image(path, self._lbl_gallery_img)

    def _display_image(self, path: str, label: QLabel):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    # ---- Helpers -----------------------------------------------------------

    @staticmethod
    def _num_item(n: int) -> QTableWidgetItem:
        item = QTableWidgetItem(str(n))
        item.setTextAlignment(Qt.AlignCenter)
        return item

    @staticmethod
    def _score_item(score: float) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{score:.4f}")
        item.setTextAlignment(Qt.AlignCenter)
        # Color-code: green for high, red for low
        if score >= 0.5:
            item.setForeground(Qt.darkGreen)
        elif score >= 0.2:
            item.setForeground(Qt.darkYellow)
        else:
            item.setForeground(Qt.darkRed)
        return item

    # ---- Report generation ---------------------------------------------------

    def _generate_report(self):
        """Generate an HTML report with all matching results and save as PDF-ready HTML."""
        if not self._all_results:
            QMessageBox.warning(self, "No Results", "Run matching first before generating a report.")
            return

        # Ask where to save
        default_name = f"Fingerprint_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default_name,
            "HTML Files (*.html);;All Files (*)"
        )
        if not save_path:
            return

        try:
            html = self._build_report_html()
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            QMessageBox.information(
                self, "Report Saved",
                f"Report saved to:\n{save_path}\n\nOpen it in your browser and use Print → Save as PDF."
            )
            self.statusBar().showMessage(f"Report saved: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report:\n{e}")

    def _build_report_html(self) -> str:
        """Build a complete HTML report from matching results."""
        n_probes = len(self._all_results)
        elapsed = getattr(self, '_elapsed', 0.0)
        gallery_count = 0
        if self._all_results:
            first_key = next(iter(self._all_results))
            gallery_count = len(self._all_results[first_key])

        # Compute CMC metrics
        # For each probe, find rank of correct match (using naming convention)
        ranks = []
        for probe_name, results in self._all_results.items():
            correct_rank = self._find_correct_rank(probe_name, results)
            if correct_rank is not None:
                ranks.append(correct_rank)

        n_with_gt = len(ranks)
        rank1 = sum(1 for r in ranks if r <= 1) if ranks else 0
        rank5 = sum(1 for r in ranks if r <= 5) if ranks else 0
        rank10 = sum(1 for r in ranks if r <= 10) if ranks else 0
        rank20 = sum(1 for r in ranks if r <= 20) if ranks else 0
        rank50 = sum(1 for r in ranks if r <= 50) if ranks else 0
        rank100 = sum(1 for r in ranks if r <= 100) if ranks else 0
        mean_rank = sum(ranks) / len(ranks) if ranks else 0
        median_rank = sorted(ranks)[len(ranks) // 2] if ranks else 0
        best_rank = min(ranks) if ranks else 0
        worst_rank = max(ranks) if ranks else 0

        # Per-probe results table rows
        probe_rows = ""
        for probe_name in sorted(self._all_results.keys()):
            results = self._all_results[probe_name]
            if not results:
                continue
            top = results[0]
            correct_rank = self._find_correct_rank(probe_name, results)
            rank_str = str(correct_rank) if correct_rank else "N/A"
            color = "#28a745" if correct_rank and correct_rank == 1 else (
                "#ffc107" if correct_rank and correct_rank <= 10 else "#dc3545"
            )
            probe_rows += f"""
            <tr>
                <td>{probe_name}</td>
                <td style="color:{color}; font-weight:bold;">{rank_str}</td>
                <td>{top['gallery_id']}</td>
                <td>{top.get('fused_score', 0):.4f}</td>
                <td>{top.get('minutiae_score', 0):.4f}</td>
                <td>{top.get('texture_score', 0):.4f}</td>
                <td>{top.get('sift_score', 0):.4f}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fingerprint Matching Report</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; line-height: 1.6; }}
    h1 {{ color: #1a5276; border-bottom: 3px solid #2a7de1; padding-bottom: 10px; }}
    h2 {{ color: #2a7de1; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
    .summary-box {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
    .metric-card {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; text-align: center; }}
    .metric-card .value {{ font-size: 2em; font-weight: bold; color: #2a7de1; }}
    .metric-card .label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
    th {{ background: #2a7de1; color: white; padding: 10px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #dee2e6; }}
    tr:hover {{ background: #f5f5f5; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .rank1 {{ background: #d4edda !important; }}
    .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #888; font-size: 0.85em; }}
    .cmc-bar {{ height: 24px; background: #2a7de1; border-radius: 4px; color: white; font-size: 0.8em;
                display: flex; align-items: center; padding-left: 8px; margin: 4px 0; }}
    .cmc-row {{ display: flex; align-items: center; margin: 4px 0; }}
    .cmc-label {{ width: 90px; font-weight: bold; font-size: 0.9em; }}
    .cmc-bar-bg {{ flex: 1; background: #eee; border-radius: 4px; height: 24px; }}
    @media print {{ body {{ margin: 20px; }} .metric-card {{ break-inside: avoid; }} }}
</style>
</head>
<body>

<h1>Latent Fingerprint Recognition Report</h1>
<p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>

<div class="summary-box">
    <strong>Evaluation Summary:</strong> {n_probes} probes matched against {gallery_count} gallery images
    in {elapsed:.1f}s ({elapsed/max(n_probes,1):.1f}s per probe).
    Ground truth detected for {n_with_gt} probes.
</div>

<h2>CMC Accuracy</h2>
<div class="metric-grid">
    <div class="metric-card">
        <div class="value">{rank1}/{n_with_gt} ({100*rank1/max(n_with_gt,1):.1f}%)</div>
        <div class="label">Rank-1 Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="value">{rank5}/{n_with_gt} ({100*rank5/max(n_with_gt,1):.1f}%)</div>
        <div class="label">Rank-5 Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="value">{rank10}/{n_with_gt} ({100*rank10/max(n_with_gt,1):.1f}%)</div>
        <div class="label">Rank-10 Accuracy</div>
    </div>
</div>

<h2>CMC Curve</h2>
<div class="cmc-row"><span class="cmc-label">Rank-1</span><div class="cmc-bar-bg"><div class="cmc-bar" style="width:{100*rank1/max(n_with_gt,1):.1f}%">{100*rank1/max(n_with_gt,1):.1f}%</div></div></div>
<div class="cmc-row"><span class="cmc-label">Rank-5</span><div class="cmc-bar-bg"><div class="cmc-bar" style="width:{100*rank5/max(n_with_gt,1):.1f}%">{100*rank5/max(n_with_gt,1):.1f}%</div></div></div>
<div class="cmc-row"><span class="cmc-label">Rank-10</span><div class="cmc-bar-bg"><div class="cmc-bar" style="width:{100*rank10/max(n_with_gt,1):.1f}%">{100*rank10/max(n_with_gt,1):.1f}%</div></div></div>
<div class="cmc-row"><span class="cmc-label">Rank-20</span><div class="cmc-bar-bg"><div class="cmc-bar" style="width:{100*rank20/max(n_with_gt,1):.1f}%">{100*rank20/max(n_with_gt,1):.1f}%</div></div></div>
<div class="cmc-row"><span class="cmc-label">Rank-50</span><div class="cmc-bar-bg"><div class="cmc-bar" style="width:{100*rank50/max(n_with_gt,1):.1f}%">{100*rank50/max(n_with_gt,1):.1f}%</div></div></div>
<div class="cmc-row"><span class="cmc-label">Rank-100</span><div class="cmc-bar-bg"><div class="cmc-bar" style="width:{100*rank100/max(n_with_gt,1):.1f}%">{100*rank100/max(n_with_gt,1):.1f}%</div></div></div>

<h2>Statistics</h2>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Probes Evaluated</td><td>{n_probes}</td></tr>
    <tr><td>Gallery Size</td><td>{gallery_count}</td></tr>
    <tr><td>Probes with Ground Truth</td><td>{n_with_gt}</td></tr>
    <tr><td>Mean Rank</td><td>{mean_rank:.1f}</td></tr>
    <tr><td>Median Rank</td><td>{median_rank}</td></tr>
    <tr><td>Best Rank</td><td>{best_rank}</td></tr>
    <tr><td>Worst Rank</td><td>{worst_rank}</td></tr>
    <tr><td>Total Time</td><td>{elapsed:.1f}s ({elapsed/max(n_probes,1):.1f}s per probe)</td></tr>
</table>

<h2>Per-Probe Results</h2>
<table>
    <tr>
        <th>Probe</th><th>Correct Rank</th><th>Top-1 Gallery</th>
        <th>Fused</th><th>Minutiae</th><th>Texture</th><th>SIFT</th>
    </tr>
    {probe_rows}
</table>

<div class="footer">
    <p>Fingerprint Matcher v2 — SIFT + Minutiae + Texture Fusion</p>
    <p>Report generated automatically by the matching GUI.</p>
</div>

</body>
</html>"""
        return html

    def _find_correct_rank(self, probe_name: str, results: list) -> int | None:
        """Find the rank of the correct gallery match using naming convention."""
        import re
        match = re.match(r"(\d+)_\d+", probe_name)
        if not match:
            return None
        subject_num = match.group(1)

        # Determine folder prefix from probe path
        folder_name = None
        if self._probe_single:
            folder_name = Path(self._probe_single).parent.name
        elif self._probe_dir:
            folder_name = Path(self._probe_dir).name

        # Try matching gallery IDs
        for i, r in enumerate(results):
            gid = r.get("gallery_id", "")
            # Check if gallery ID matches: folder_subject_ pattern
            if folder_name and gid.startswith(f"{folder_name}_{subject_num}_"):
                return i + 1
            # Fallback: just check subject number pattern
            if re.search(rf"[_a]{subject_num}_\d+$", gid):
                return i + 1

        return None