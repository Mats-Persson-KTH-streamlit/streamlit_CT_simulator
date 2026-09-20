"""Entry point: layout, widget wiring, orchestration.

This is the only module that lays out the page or decides what gets
rendered where; every other module is either pure (ct_model, plotting) or
only touches session_state through the keys defined in state.py.
"""

import matplotlib.pyplot as plt
import streamlit as st

import config
import ct_model
import drawing
import plotting
import scan
import state


def _refresh_image_and_projection(image_placeholder, proj_placeholder):
    """Redraws the gantry-sketch panel and the p(t) plot from whatever's
    currently in session_state (image_hu, theta_deg).

    Shared by render_reactive_panels (triggered by the angle slider) and
    render_canvas_and_brush_panel (triggered by a brush stroke) -- both need
    the exact same "recompute p(t), redraw both panels" step, just triggered
    by a different input changing.
    """
    image_hu = st.session_state[state.IMAGE_HU]
    theta_deg = st.session_state[state.THETA_DEG]
    p_t = ct_model.project_single_angle(image_hu, theta_deg)

    fig_img = plotting.render_image_panel(image_hu, theta_deg)
    image_placeholder.pyplot(fig_img, clear_figure=True, width=config.IMAGE_SIZE_PX)
    plt.close(fig_img)

    fig_p = plotting.render_projection_plot(p_t)
    proj_placeholder.pyplot(fig_p, clear_figure=True, width='stretch')
    plt.close(fig_p)


@st.fragment
def render_reactive_panels(image_placeholder, proj_placeholder, sino_placeholder):
    """The angle slider, plus everything that depends on angle or image_hu:
    the gantry sketch, p(t) plot, and sinogram dashed line.

    Grouping these in one fragment means dragging the slider only reruns
    this function -- Streamlit skips the rest of the script entirely, so the
    drawing canvas and brush controls (outside this fragment) are never
    re-sent to the browser and never flicker just because the angle changed.
    The slider itself is rendered with a bare st.slider() call (no
    `with col_left:` wrapper): Streamlit forbids a fragment from placing a
    *widget* in a container outside the fragment's own position, so this
    fragment must itself be called from within `with col_left:` in the main
    script -- the slider then lands in col_left simply because that's this
    fragment's own spot.

    This fragment must NEVER be merged with render_canvas_and_brush_panel
    (tried once): even though the canvas's own arguments end up
    byte-identical across an angle-only rerun, streamlit-drawable-canvas's
    frontend still visibly reinitializes itself whenever it's part of a
    fragment's re-executed subtree, regardless of whether its props actually
    changed. Keeping the canvas out of this fragment entirely is what keeps
    dragging the angle slider glitch-free.

    All three panels are filled via pre-existing st.empty() placeholders
    (created once by the caller, in their correct layout positions) rather
    than fresh `with col_x: st.pyplot(...)` calls made from inside here.
    That distinction matters: a fragment only knows how to clear its *own*
    reserved region of the page before re-rendering -- writing fresh
    elements into someone else's container (col_right) on a fragment-scoped
    rerun doesn't clear anything there first, so each rerun just appended
    another pair of plots below the last (the "growing stack of plots" bug).
    Calling .pyplot() on an existing placeholder always replaces that exact
    slot in place, on any kind of rerun, avoiding that entirely -- the same
    trick is what lets render_canvas_and_brush_panel (a different fragment)
    also write into image_placeholder/proj_placeholder without owning them.

    Deliberately does NOT take a `scan_clicked` argument. A fragment's
    arguments get captured in a closure that's stored and reused on every
    later *fragment-scoped* rerun (e.g. dragging the slider) -- if this
    function were only called with scan_clicked=True during the scan run,
    that True would be baked into the stored closure forever, and every
    subsequent slider-only rerun would silently early-return and never
    redraw again (exactly the "dashed line stops responding after a scan"
    bug). Always redrawing unconditionally, from fresh session_state, avoids
    depending on any value that isn't safe to have gone stale. When a scan
    actually happens, scan.run_scan() (called right after this, in the main
    script) simply overwrites the same placeholders again immediately.
    """
    st.slider(
        "Gantry angle (deg)",
        min_value=0.0,
        max_value=360.0,
        # Explicitly seeded from session_state -- see the identical, more
        # visible fix in drawing.render_hu_slider_and_colorbar() for why the
        # no-value fallback isn't safe here either. Invisible today only
        # because the default (0) happens to equal min_value.
        value=float(st.session_state[state.THETA_DEG]),
        step=1.0,
        key=state.THETA_DEG,
    )

    _refresh_image_and_projection(image_placeholder, proj_placeholder)

    fig_s = plotting.render_sinogram(st.session_state[state.SINOGRAM], st.session_state[state.THETA_DEG])
    sino_placeholder.pyplot(fig_s, clear_figure=True, width='stretch')
    plt.close(fig_s)


@st.fragment
def render_canvas_and_brush_panel(image_placeholder, proj_placeholder):
    """The drawing canvas and the brush-size/HU controls that determine its
    stroke_width/stroke_color, plus the panels that depend on image_hu.

    These MUST be one fragment, not two -- a brush-size or brush-color
    change has to reach the canvas's own stroke_width/stroke_color arguments
    before the *next* stroke, but Streamlit only reruns the fragment a
    widget itself is declared in. A fragment scoped to only the brush
    radio/slider would update session_state[BRUSH_*] when they change, but
    couldn't force a *separate* fragment holding the canvas to rerun and
    pick up the new value -- that fragment would only rerun the next time
    ITS OWN widget (the canvas) fires, i.e. one stroke too late.

    Called from `with col_mid:` in the main script (not col_left) --
    Streamlit fragments can only place *widgets* inside containers they
    create themselves (confirmed empirically:
    `StreamlitFragmentWidgetsNotAllowedOutsideError: Fragments cannot write
    widgets to outside containers`, even via a pre-existing st.empty()), and
    this fragment needs to place TWO widgets (brush controls, canvas) that
    used to live in two different page-level columns (col_left, col_mid).
    Building a brand new page-level row of columns to hold both was tried
    and rejected: a fresh st.columns() row always starts below the *entire*
    previous row, at the height of that row's tallest column, which left an
    unavoidable gap either above the brush panel or above the canvas
    depending on which row absorbed the tall image/proj/sino placeholders.

    Instead, both widgets are nested INSIDE col_mid, which this fragment
    already owns (it's invoked from `with col_mid:`) -- avoiding a new
    *page*-level row entirely. canvas_col comes FIRST (left) so the canvas
    stays flush with col_mid's left edge, directly under image_placeholder
    (also unindented, so it's pinned to that same edge) -- keeping "gantry
    sketch, then drawing area" vertically aligned as one visual column,
    exactly as before. brush_col is a narrow strip to its right -- the one
    real trade-off of this approach: the brush panel/HU slider/colorbar move
    out of col_left (where the spec originally put them, under the angle
    slider) into this strip beside the canvas, because col_mid is the only
    container this fragment is allowed to place widgets in without
    triggering the row-break problem above.

    Refreshes image_placeholder/proj_placeholder itself after a stroke
    (plain elements, not widgets -- allowed even though image_placeholder
    was created outside this fragment, by the same rule that lets
    render_reactive_panels write into sino_placeholder). Deliberately never
    touches sino_placeholder -- drawing doesn't change the already-computed
    sinogram, only a new Scan does.
    """
    canvas_col, brush_col = st.columns([3, 1])

    with canvas_col:
        canvas_result = drawing.render_canvas()
        drawing.apply_paint_from_canvas(canvas_result)

    with brush_col:
        drawing.render_brush_panel()
        drawing.render_hu_slider_and_colorbar()

    _refresh_image_and_projection(image_placeholder, proj_placeholder)


st.set_page_config(page_title="CT Simulator", layout="wide")
state.init_state()

col_left, col_mid, col_right = st.columns([1, 2, 2])  # ~20% / 40% / 40%

with col_left:
    st.title("Parallel-Beam CT Simulator")
    st.caption(
        "Pick a CT number and brush size, then draw inside the circle. "
        "Use the angle slider to rotate the detector manually. "
        "Click Scan to acquire the full sinogram."
    )
    scan_clicked = st.button("Scan", type="primary")

with col_mid:
    image_placeholder = st.empty()

with col_right:
    proj_placeholder = st.empty()
    sino_placeholder = st.empty()

with col_left:
    render_reactive_panels(image_placeholder, proj_placeholder, sino_placeholder)

with col_mid:
    render_canvas_and_brush_panel(image_placeholder, proj_placeholder)

if scan_clicked:
    scan.run_scan({"image": image_placeholder, "projection": proj_placeholder, "sinogram": sino_placeholder})
