"""Step 7 — interactive and publication heatmaps of isotopologue-channel intensities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import design as dz, heatmap as hm, plots as pl
from ui import common as C


def _col_info(D, cols):
    """Annotation table for the heatmap columns (samples): group, time, treatment, replicate, metadata columns."""
    s = C.ss()
    I = D.set_index("Sample")
    extra = [c for c in s.ds.S.columns if c.startswith("Meta: ")]
    if extra:
        I = I.join(s.ds.S.set_index("Sample")[extra])
    I = I.loc[cols]
    return I.assign(Time=I.TimeLabel)


def render():
    C.header(7, "Heatmaps",
             "Clustered heatmaps of individual isotopologue channels (active intensity dataset from the sidebar, row-wise "
             "Z-score, one column per sample). Rows can be selected by p value or FDR from a Step 6 result.")
    if not C.require("upload", "design", "normalize"):
        return
    s = C.ss()
    D = C.design_final()
    An = dz.analysed(D)
    arms = dz.groups_of(D)
    X, info = C.layer("int")
    M = X.copy()

    with st.expander("Rows to show", expanded=True):
        r = st.columns([2, 2, 1, 1])
        mode = r[0].radio("Select rows by", ["sig", "var", "all"], key="hm_rows",
                          format_func={"sig": "Significance (p value / FDR, Step 6)", "var": "Top N by variance",
                                       "all": "All channels"}.get)
        if mode == "sig":
            if not s.stats:
                r[1].warning("Run Step 6 first; showing top 60 by variance.")
                mode = "var"
            else:
                rs = r[1].selectbox("Statistics result", list(s.stats), index=len(s.stats) - 1, key="hm_res")
                T = s.stats[rs]["table"]
                comps = list(dict.fromkeys(T.Comparison))
                csel = r[1].multiselect("Comparisons (significant in any)", comps, default=comps, key="hm_cmps")
                crit = r[2].radio("Criterion", ["FDR", "P value", "P value and FDR"], key="hm_crit")
                thr_q = r[3].number_input("FDR threshold (≤ 1)", 0.0001, 1.0, 0.25, 0.01, format="%.4f",
                                          key="hm_thr_fdr", disabled=crit == "P value")
                thr_p = r[3].number_input("P value threshold (≤ 1)", 0.0001, 1.0, 0.05, 0.01, format="%.4f",
                                          key="hm_thr_p", disabled=crit == "FDR")
                TT = T[T.Comparison.isin(csel)]
                qc = ["FDR"] if "FDR" in TT else [x for x in TT.columns if x.startswith("FDR ")]
                pc_ = ["p"] if "p" in TT else [x for x in TT.columns if x.startswith("p ")]
                if crit == "FDR":
                    hit = (TT[qc] <= thr_q).any(axis=1)
                    thr_txt = f"FDR ≤ {thr_q:g}"
                elif crit == "P value":
                    hit = (TT[pc_] <= thr_p).any(axis=1)
                    thr_txt = f"p ≤ {thr_p:g}"
                else:
                    hit = (TT[pc_] <= thr_p).any(axis=1) & (TT[qc] <= thr_q).any(axis=1)
                    thr_txt = f"p ≤ {thr_p:g} and FDR ≤ {thr_q:g}"
                sig = TT[hit].Feature.unique()
                Ms = M.loc[[f for f in M.index if f in set(sig)]]
                if len(Ms) >= 2:
                    M = Ms
                    st.caption(f"{len(M)} channels with {thr_txt} in at least one selected comparison of "
                               f"“{rs}”.")
                else:
                    C.note(f"Only {len(Ms)} channel(s) have {thr_txt} in “{rs}”; a heatmap needs ≥ 2 rows. "
                           "Showing the 60 most variable channels instead — try “P value” or a higher threshold.")
                    M = M.loc[M.var(axis=1).sort_values(ascending=False).index[:60]]
        if mode == "var":
            top = r[1].number_input("Top N rows by variance", 2, 5000, 60, 10, key="hm_top")
            M = M.loc[M.var(axis=1).sort_values(ascending=False).index[:int(top)]]
        if not r[1].toggle("Include M+0 channels", True, key="hm_m0"):
            M = M[~M.index.str.endswith("| M+0")]
        pick = st.multiselect("Highlight rows", sorted(M.index), key="hm_pick")
    M = M.dropna(how="all")
    if len(M) < 2:
        st.warning("Fewer than two rows match the selection."); return

    ci = _col_info(D, list(M.columns))
    base_tracks = [t for t in ["Group", "Time", "Treatment", "Replicate"] if t in ci.columns
                   and ci[t].astype(str).str.strip().replace({"0": "", "nan": ""}).ne("").any()]
    meta_tracks = [t for t in ci.columns if t.startswith("Meta: ")]
    with st.expander("Column annotations, clustering and colours", expanded=False):
        tracks = st.multiselect("Column annotations (sample group and metadata)", base_tracks + meta_tracks,
                                default=[t for t in ["Group", "Time"] if t in base_tracks], key="hm_tracks")
        z = st.columns(3)
        cr_ = z[0].toggle("Cluster rows", True, key="hm_cr")
        cc_ = z[1].toggle("Cluster columns", False, key="hm_cc")
        row_labels = z[2].toggle("Row labels", True, key="hm_rl")
        k = st.columns(3)
        metric = k[0].selectbox("Distance", list(hm.DISTANCES), format_func=hm.DISTANCES.get, key="hm_dist")
        method = k[1].selectbox("Linkage", list(hm.LINKAGES), format_func=hm.LINKAGES.get, key="hm_link")
        cmap = k[2].selectbox("Colour scale", list(hm.CMAPS), key="hm_cmap")
        ap = st.columns(3)
        auto = ap[0].toggle("Automatic colour limits", True, key="hm_auto")
        lim = ap[1].number_input("Colour limit ±", 0.1, 100.0, 3.0, 0.5, key="hm_lim", disabled=auto)
    n_cols = M.shape[1]
    lab_len = max((len(str(r)) for r in M.index), default=10)
    with st.expander("Figure size and fonts", expanded=False):
        f0 = st.columns(5)
        cell_w = f0[0].number_input("Cell width (in)", 0.08, 0.80, 0.30, 0.02, format="%.2f", key="hm_cell_w",
                                     help="Width of one sample column. Default 0.30 in: a wide, landscape heatmap.")
        cell = f0[1].number_input("Cell height (in)", 0.06, 0.30, 0.10, 0.01, format="%.2f", key="hm_cell",
                                  help="Height of one row. 0.10 in keeps tall heatmaps compact; row labels are "
                                       "sized to the row (≈ 6 pt) so they stay readable; the figure size follows from the number of rows and "
                                       "samples.")
        custom = f0[2].toggle("Custom figure size", False, key="hm_fig_custom",
                              help="Off: the size follows from the cell size. On: type the exact figure size; "
                                   "cells then stretch to fill it.")
        auto_w = round(0.9 + cell_w * n_cols + 0.052 * lab_len + 1.65, 1)
        auto_h = round(max(3.0, cell * len(M) + 3.0), 1)
        fig_w = f0[3].number_input("Figure width (in)", 3.0, 40.0, auto_w, 0.5, key="hm_fig_w", disabled=not custom,
                                   help="Used for the heatmap on screen (96 px per inch) and for the download box.")
        fig_h = f0[4].number_input("Figure height (in)", 3.0, 200.0, auto_h, 0.5, key="hm_fig_h",
                                   disabled=not custom)
        if not custom:
            fig_w, fig_h = auto_w, auto_h
        f1 = st.columns(4)
        fam = f1[3].selectbox("Font family", hm.FONT_FAMILIES, key="hm_font_family")
        f2 = st.columns(5)
        fonts = dict(
            family=fam,
            title=f2[0].number_input("Title (pt)", 4.0, 30.0, hm.DEFAULT_FONTS["title"], 0.5, key="hm_fs_title"),
            row=f2[1].number_input("Row labels (pt, 0 = auto-fit)", 0.0, 24.0, hm.DEFAULT_FONTS["row"], 0.5,
                                   key="hm_fs_row"),
            col=f2[2].number_input("Column labels (pt)", 3.0, 24.0, hm.DEFAULT_FONTS["col"], 0.5, key="hm_fs_col"),
            ann=f2[3].number_input("Annotations & legend (pt)", 3.0, 24.0, hm.DEFAULT_FONTS["ann"], 0.5,
                                   key="hm_fs_ann"),
            scale=f2[4].number_input("Colour scale (pt)", 3.0, 24.0, hm.DEFAULT_FONTS["scale"], 0.5,
                                     key="hm_fs_scale"))
        st.caption("Sizes in points apply to the exported figure; the on-screen heatmap uses the same family and "
                   "proportional sizes. Label space grows with the font size, so text never overlaps.")
    # keep the download box in step with the size chosen here (only when this size changes)
    if s.get("_hm_size_prev") != (fig_w, fig_h):
        st.session_state["exp_hm_w"], st.session_state["exp_hm_h"] = float(fig_w), float(fig_h)
        st.session_state["_hm_size_prev"] = (fig_w, fig_h)
    Z, diverging, label = hm.zscore(M, "row", None), True, "Row Z-score"
    Z = Z.dropna(how="all")
    vmin = vmax = None
    if not auto:
        vmin, vmax = -lim, lim
    if not cc_:
        order = An.sort_values(["Group", "Hours"], key=lambda x: x.map({g: i for i, g in enumerate(arms)})
                               if x.name == "Group" else x).Sample
        Z = Z[[x for x in order if x in Z.columns]]
    ci = ci.loc[Z.columns]
    s.results["heatmap_done"] = True
    title = f"Isotopologue channels — {info['unit']}, row Z-score"
    Zf = Z.fillna(0)
    ro = hm.cluster_order(Zf, metric, method)[0] if cr_ and len(Z) > 2 else list(range(len(Z)))
    co = hm.cluster_order(Zf.T, metric, method)[0] if cc_ and Z.shape[1] > 2 else list(range(Z.shape[1]))
    ZZ = Z.iloc[ro, co]
    _show_sized(_plotly(ZZ, ci.iloc[co], tracks, diverging, vmin, vmax, label, cmap, row_labels, pick, title,
                        arms, fonts, fig_w, fig_h, cell_px=None if custom else cell_w * PX_PER_IN,
                        cell_h_px=None if custom else cell * PX_PER_IN))
    st.caption(f"{len(ZZ)} rows × {ZZ.shape[1]} samples. Row-wise Z-score of {info['unit']}"
               + (f"; rows clustered with {hm.DISTANCES[metric]} distance and {hm.LINKAGES[method]} linkage."
                  if cr_ else "."))

    def make(w=11.0, h=None):
        # size unchanged from the automatic one → exact cell size; otherwise stretch to the typed size
        sized = custom or abs(w - fig_w) > 0.05 or abs((h or fig_h) - fig_h) > 0.05
        kw = dict(width=w, height=h) if sized else dict(width=w, height=None, cell_w=cell_w, cell_h_in=cell)
        f, _, _ = hm.draw_heatmap(Z.fillna(0) if (cr_ or cc_) else Z, ci, title=title, value_label=label,
                                  diverging=diverging, vmin=vmin, vmax=vmax, cluster_rows=cr_, cluster_cols=cc_,
                                  metric=metric, method=method, row_labels=row_labels, **kw,
                                  group_order=arms, highlight=pick, annotations=tracks, cmap=cmap, fonts=fonts)
        return f
    C.export_widget(lambda w, h: make(w, h), "heatmap_isotopologue_channels", "exp_hm",
                    (float(fig_w), float(fig_h)), caption=f"Heatmap — {title}",
                    tables={"Z-score matrix (as plotted)": ZZ.reset_index().rename(columns={"index": "Channel"}),
                            f"{info['unit']} values of these rows": M.loc[ZZ.index].reset_index().rename(
                                columns={"index": "Channel"})})
    C.next_button("pathways", "Pathway mapping")


PX_PER_IN = 96
PT = 1.4          # on-screen px per exported pt (keeps the screen and the export in proportion)


def _show_sized(fig):
    """Show the heatmap at the chosen pixel width (falls back to the container width on old Streamlit)."""
    cfg = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 5}, "scrollZoom": True}
    w, h = int(fig.layout.width), int(fig.layout.height)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    for kw in (dict(width=w, height=h), dict(use_container_width=False)):
        try:
            return st.plotly_chart(fig, key="hm_plotly", config=cfg, theme=None, **kw)
        except Exception:
            continue
    return st.plotly_chart(fig, key="hm_plotly", config=cfg, theme=None)


def _plotly(ZZ, ci, tracks, diverging, vmin, vmax, label, cmap, row_labels, pick, title, arms, fonts=None,
            width_in=11.0, height_in=None, cell_px=None, cell_h_px=None):
    fo = {**hm.DEFAULT_FONTS, **(fonts or {})}
    fam = f"{fo['family']}, Liberation Sans, Helvetica, sans-serif"
    nt = len(tracks)
    nrows = len(ZZ)
    if cell_px:   # fixed cell size; a fixed row-label font keeps its size and makes the rows taller if needed
        ch = cell_h_px or cell_px
        row_px = (fo["row"] * PT) if fo["row"] else max(7.5, min(10.0, ch * 0.85))
        cell_h_px = max(ch, row_px * 1.12)
    else:
        row_px = (fo["row"] * PT) if fo["row"] else 10
    col_px, ann_px, sc_px, ti_px = fo["col"] * PT, fo["ann"] * PT, fo["scale"] * PT, fo["title"] * PT
    h_tr = max(16, int(ann_px * 1.5))
    xlab = int(max(len(str(c)) for c in ZZ.columns) * col_px * 0.55 + 12)
    leg_w = int(max([len("Time" if t == "TimeLabel" else t) for t in tracks] +
                    [len(str(v)) for t in tracks for v in ci[t].unique()] + [4]) * ann_px * 0.52 + 30) if tracks else 10
    rlab = int(max(len(str(r)) for r in ZZ.index) * row_px * 0.5 + 8) if row_labels else 6
    cb_title_px = int(sc_px * 1.6 + 6)           # colour-scale title (its own line, above the bar)
    cb_px = int(cb_title_px + 14 + sc_px * 1.6 + 16)   # title + bar + tick labels underneath the heatmap
    t_px = int(ti_px * 2.2 + 20)
    if cell_px:
        h_main = int(round(cell_h_px * nrows))
    elif height_in:
        total = int(height_in * PX_PER_IN)
        h_main = max(total - h_tr * nt - xlab - cb_px - t_px - 20, 120)
    else:
        h_main = max(380, int(max(15, row_px * 1.5) * nrows) + 120)
    width_px = int(width_in * PX_PER_IN)
    if cell_px:   # plot area = exactly n_cols cells; figure grows to fit the title and the colour scale
        plot_w = int(round(cell_px * len(ZZ.columns)))
        width_px = max(60 + plot_w + rlab + leg_w + 10, int(len(title) * ti_px * 0.55) + 40, 60 + 140 + rlab + leg_w)
    fig = make_subplots(rows=nt + 1, cols=1, shared_xaxes=True, vertical_spacing=0.004,
                        row_heights=[h_tr] * nt + [h_main])
    cols = list(ZZ.columns)
    acol = hm.annotation_colors(ci, tracks, arms)
    for k, t in enumerate(tracks):
        vals = list(ci[t])
        cats = list(acol[t])
        code = [cats.index(v) if v in cats else np.nan for v in vals]
        n = max(len(cats), 1)
        scale = []
        for i, cv in enumerate(cats):
            scale += [[i / n, acol[t][cv]], [(i + 1) / n, acol[t][cv]]]
        fig.add_trace(go.Heatmap(z=[code], x=cols, y=["Time" if t == "TimeLabel" else t], zmin=-0.5, zmax=n - 0.5,
                                 colorscale=scale or [[0, "#eee"], [1, "#eee"]], showscale=False,
                                 customdata=[[str(v) for v in vals]],
                                 hovertemplate="%{x}<br>" + t + ": %{customdata}<extra></extra>"), row=k + 1, col=1)
    if cmap in (None, "Omnix diverging"):
        cs = pl.plotly_div_scale() if diverging else pl.plotly_seq_scale()
    else:
        cs = hm.PLOTLY_SCALE.get(cmap, cmap)
    lim_ = float(np.nanmax(np.abs(ZZ.values))) if vmax is None else vmax
    dom = fig.layout[f"yaxis{nt + 1}" if nt else "yaxis"].domain or (0, 1)
    dom_mid, dom_len = (dom[0] + dom[1]) / 2, dom[1] - dom[0]
    fig.add_trace(go.Heatmap(z=ZZ.values, x=cols, y=list(ZZ.index), colorscale=cs, zmid=0 if diverging else None,
                             zmin=-lim_ if diverging else vmin, zmax=lim_ if diverging else vmax,
                             colorbar=dict(orientation="h", thickness=14, len=120, lenmode="pixels",
                                           x=1, xanchor="right", xpad=0, yanchor="top",
                                           y=-(xlab + cb_title_px) / (h_main + h_tr * nt + 20), ypad=0, tickfont=dict(size=max(sc_px - 1, 6), color="black"),
                                           nticks=5, outlinecolor="black", outlinewidth=0.5),
                             hovertemplate="%{y}<br>%{x}<br>%{z:.3g}<extra></extra>"), row=nt + 1, col=1)
    fig.update_yaxes(autorange="reversed", showticklabels=row_labels, tickfont=dict(size=row_px, color="black"),
                     tickmode="array", tickvals=list(ZZ.index), ticktext=[str(r) for r in ZZ.index],
                     side="right", row=nt + 1, col=1)
    for k in range(nt):
        fig.update_yaxes(tickfont=dict(size=ann_px, color="black"), row=k + 1, col=1)
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=col_px, color="black"), tickmode="array", tickvals=cols,
                     ticktext=[str(c) for c in cols], row=nt + 1, col=1)
    for rname in ZZ.index:
        if rname in pick:
            fig.add_annotation(x=1.0, xref="paper", y=rname, yref=f"y{nt + 1}" if nt else "y", text="◀",
                               showarrow=False, xanchor="left", xshift=rlab, font=dict(color="#a8402c"))
    # annotation legend: right side, vertically centred on the heatmap
    leg = "<br><br>".join(f"<b>{'Time' if t == 'TimeLabel' else t}</b><br>" + "<br>".join(
        f"<span style='color:{c}'>■</span> {v:g}" if isinstance(v, float) else f"<span style='color:{c}'>■</span> {v}"
        for v, c in acol[t].items()) for t in tracks)
    cb_title = dict(text=label, x=1, xref="paper", xanchor="right", y=0, yref="paper", yanchor="top",
                    yshift=-(xlab + 2), showarrow=False, font=dict(size=sc_px, color="black"))
    l_px = max(60, width_px - plot_w - (rlab + leg_w + 10)) if cell_px else 60
    n_leg = sum(1 + len(acol[t]) for t in tracks) + max(len(tracks) - 1, 0)
    extra_b = max(0, int(n_leg * ann_px * 1.3) - (h_main + xlab + cb_px)) if tracks else 0
    fig.update_layout(template="plotly_white", width=width_px,
                      height=h_main + h_tr * nt + xlab + cb_px + t_px + 20 + extra_b,
                      font=dict(family=fam, color="black"),
                      title=dict(text=title, x=0.5, xanchor="center",
                                 font=dict(size=ti_px, family=fam, color="black")),
                      margin=dict(l=l_px, r=rlab + leg_w + 10, t=t_px, b=xlab + cb_px + extra_b),
                      annotations=list(fig.layout.annotations) + [cb_title] + ([dict(text=leg, x=1.0, xref="paper", xanchor="left",
                                                                         xshift=rlab + 6,
                                                                         y=dom_mid, yref="paper", yanchor="middle",
                                                                         showarrow=False, align="left",
                                                                         font=dict(size=ann_px, color="black"))]
                                                                   if tracks else []))
    return fig
