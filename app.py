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


@st.fragment
def render_reactive_panels(image_placeholder, proj_placeholder, sino_placeholder):
    """The angle slider, plus everything that depends on angle or image_hu:
    the gantry sketch, p(t) plot, and sinogram dashed line.

    Grouping these in one fragment means dragging the slider only reruns
    this function -- Streamlit skips the rest of the script entirely, so the
    drawing canvas (outside this fragment) is never re-sent to the browser
    and never flickers just because the angle changed. The slider itself is
    rendered with a bare st.slider() call (no `with col_left:` wrapper):
    Streamlit forbids a fragment from placing a *widget* in a container
    outside the fragment's own position, so this fragment must itself be
    called from within `with col_left:` in the main script -- the slider
    then lands in col_left simply because that's this fragment's own spot.

    All three panels are filled via pre-existing st.empty() placeholders
    (created once by the caller, in their correct layout positions) rather
    than fresh `with col_x: st.pyplot(...)` calls made from inside here.
    That distinction matters: a fragment only knows how to clear its *own*
    reserved region of the page before re-rendering -- writing fresh
    elements into someone else's container (col_right) on a fragment-scoped
    rerun doesn't clear anything there first, so each rerun just appended
    another pair of plots below the last (the "growing stack of plots" bug).
    Calling .pyplot() on an existing placeholder always replaces that exact
    slot in place, on any kind of rerun, avoiding that entirely.

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

    image_hu = st.session_state[state.IMAGE_HU]
    theta_deg = st.session_state[state.THETA_DEG]
    p_t = ct_model.project_single_angle(image_hu, theta_deg)

    fig_img = plotting.render_image_panel(image_hu, theta_deg)
    image_placeholder.pyplot(fig_img, clear_figure=True, width=config.IMAGE_SIZE_PX)
    plt.close(fig_img)

    fig_p = plotting.render_projection_plot(p_t)
    proj_placeholder.pyplot(fig_p, clear_figure=True, width='stretch')
    plt.close(fig_p)

    fig_s = plotting.render_sinogram(st.session_state[state.SINOGRAM], theta_deg)
    sino_placeholder.pyplot(fig_s, clear_figure=True, width='stretch')
    plt.close(fig_s)


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
    canvas_result = drawing.render_canvas()
    drawing.apply_paint_from_canvas(canvas_result)

with col_right:
    proj_placeholder = st.empty()
    sino_placeholder = st.empty()

with col_left:
    render_reactive_panels(image_placeholder, proj_placeholder, sino_placeholder)
    drawing.render_brush_panel()
    drawing.render_hu_slider_and_colorbar()

if scan_clicked:
    scan.run_scan({"image": image_placeholder, "projection": proj_placeholder, "sinogram": sino_placeholder})
