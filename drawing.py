"""Canvas-based painting: HU/brush-size controls plus the drawing canvas."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.elements.image as _st_image_module

import config
import ct_model
import state


def _patch_image_to_url_compat():
    """streamlit-drawable-canvas (last released 2023) calls the pre-1.40
    `streamlit.elements.image.image_to_url(image, width, clamp, channels,
    output_format, image_id)`, which Streamlit 1.57 removed.

    A first attempt returned a self-contained `data:` URI to sidestep
    Streamlit's MediaFileManager lifecycle entirely. That turned out to be
    silently, permanently broken rather than just racy: the component's own
    (compiled, unmodifiable) frontend JS unconditionally does
    `img.src = <streamlit app origin> + backgroundImageURL` -- correct for
    the relative "/media/xxxx.png"-style path this function used to return
    pre-1.40, but it turns a `data:image/png;base64,...` URI into
    `"http://host:port" + "data:image/png;base64,..."`, an invalid URL the
    browser fails to load with no visible error. That's why the canvas
    background never painted at all and the drawing area just showed
    whatever the page's own theme background was underneath.

    Delegating to the *current* `elements.lib.image_utils.image_to_url`
    (relocated, and now taking a `LayoutConfig` instead of a bare
    `width: int`) instead returns a real, relative MediaFileManager path,
    which the frontend's origin-prepending logic turns into a valid,
    loadable URL. This does reintroduce a known, narrower race: the
    MediaFileManager garbage-collects entries not "touched" by the *current*
    run, and since the canvas's background is content-addressed and re-fed
    from image_hu on every rerun, a background from immediately before the
    current stroke could in principle be GC'd out from under an in-flight
    browser fetch. That's a possible transient flicker during rapid
    continuous drawing -- clearly preferable to a background that never
    renders at all.
    """
    if hasattr(_st_image_module, "image_to_url"):
        return

    from streamlit.elements.lib.image_utils import image_to_url as _real_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
        return _real_image_to_url(image, LayoutConfig(), clamp, channels, output_format, image_id)

    _st_image_module.image_to_url = _compat_image_to_url


_patch_image_to_url_compat()

from streamlit_drawable_canvas import st_canvas  # noqa: E402  (must follow the patch above)

_BRUSH_LABELS = {
    config.BRUSH_SIZES_PX[0]: "S",
    config.BRUSH_SIZES_PX[1]: "M",
    config.BRUSH_SIZES_PX[2]: "L",
    config.BRUSH_SIZES_PX[3]: "XL",
}

CANVAS_KEY = "phantom_canvas"


def gray_u8_to_hu(gray_array):
    frac = gray_array.astype(np.float64) / 255.0
    return config.HU_MIN + frac * (config.HU_MAX - config.HU_MIN)


def hu_to_gray_hex(hu_value):
    g = int(np.clip((hu_value - config.HU_MIN) / (config.HU_MAX - config.HU_MIN) * 255, 0, 255))
    return f"#{g:02x}{g:02x}{g:02x}"


def render_brush_panel():
    """The brush-size picker (radio)."""
    st.radio(
        "Brush size",
        options=config.BRUSH_SIZES_PX,
        format_func=lambda px: _BRUSH_LABELS.get(px, str(px)),
        key=state.BRUSH_SIZE_PX,
        horizontal=True,
    )


def render_hu_slider_and_colorbar():
    """Native-widget HU picker (slider) plus the decorative gradient bar
    mirroring the mockup's HU bar, sized to match the slider above it."""
    import plotting

    st.slider(
        "CT number (HU)",
        min_value=config.HU_MIN,
        max_value=config.HU_MAX,
        # Seeded explicitly from session_state rather than left to the
        # no-value fallback (which defaults to min_value): with a
        # `@st.fragment` elsewhere on the page, Streamlit compacts session
        # state more than once per page load, and that fallback path only
        # picks up the "real" value when this key's assignment is still
        # considered "new" at the moment this widget is created -- fragile
        # timing that doesn't hold here, and the slider silently opens at
        # min_value (-1000) instead of the actual default (0).
        value=int(st.session_state[state.BRUSH_HU]),
        step=10,
        key=state.BRUSH_HU,
    )
    fig = plotting.render_hu_colorbar(st.session_state[state.BRUSH_HU])
    st.pyplot(fig, clear_figure=True, width='stretch')
    plt.close(fig)


def render_canvas():
    """Renders the paint surface and returns the raw CanvasResult.

    Always uses a fixed background_color="black", and NEVER background_image
    -- deliberately, after two rounds of trying to use background_image at
    all:

    1. Passing background_image=image_hu_to_pil(image_hu) unconditionally
       (the original design): works locally, but on Streamlit Community
       Cloud the image has been observed to simply never load (the drawing
       area shows the page's own light/dark theme color instead, and it
       never self-corrects, even after drawing) while everything else on
       the page -- including the gantry-sketch panel, which renders the
       exact same image_hu through Streamlit's own st.pyplot()/st.image()
       path rather than this component's -- displays correctly. That
       pointed at streamlit-drawable-canvas's own background-loading path
       specifically (it's unmaintained since 2023 and already has one
       documented history of origin/URL-construction bugs in this exact
       path -- see _patch_image_to_url_compat above), not at image_hu or
       the constraint pipeline -- confirmed directly, since a screenshot of
       the broken deployment still showed the correct phantom, stroke
       included, in the gantry sketch.

    2. Switching to background_color="black" only while the phantom was
       still perfectly uniform, falling back to background_image once
       anything was drawn (reasoning: a uniform image has no visual
       information a flat color can't already represent exactly, so the
       common "fresh session" case could skip the fragile path entirely).
       This introduced a WORSE bug: st_canvas's own (installed, third-party,
       unmodifiable) wrapper code -- site-packages/streamlit_drawable_canvas/
       __init__.py -- forces background_color to "" whenever background_image
       is passed at all, and its docstring explicitly warns "Changing
       background_color will reset the drawing." Switching between the two
       modes therefore toggles background_color's actual value the moment
       the phantom stopped being blank (i.e. on the user's very first
       stroke), silently wiping image_hu back to blank right as it was
       drawn on.

    Given background_image is the one proven unreliable on some
    deployments, and toggling between the two is proven to corrupt data,
    the only combination that's actually stable is to never use
    background_image and never change background_color's value at all --
    accepting the one real trade-off this leaves: the canvas no longer
    visually restores a previously-drawn phantom after a genuine page
    reload (image_hu and the gantry-sketch panel are unaffected either way,
    since they don't depend on this at all -- only the canvas's own on-screen
    appearance immediately after a reload would show blank until the next
    stroke).
    """
    stroke_color = hu_to_gray_hex(st.session_state[state.BRUSH_HU])
    stroke_width = 2 * st.session_state[state.BRUSH_SIZE_PX]

    return st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="black",
        update_streamlit=True,
        height=config.CANVAS_SIZE_PX,
        width=config.CANVAS_SIZE_PX,
        drawing_mode="freedraw",
        key=CANVAS_KEY,
    )


def apply_paint_from_canvas(canvas_result):
    """Folds the canvas' full raster back into session_state.image_hu.

    enforce_constraints() re-clips it to the circle/-1000 and HU-range
    invariants -- this is what erases any stroke that spilled past the
    circle boundary.

    On the very first render of a session, the canvas component can report
    an image_data snapshot taken before it has resized itself to its
    configured height/width (observed: (150, 300, 4) instead of
    (IMAGE_SIZE_PX, IMAGE_SIZE_PX, 4)) -- a transient glitch in the
    component's own initial layout, not a real edit. Treated as real data,
    that mismatched shape crashes enforce_constraints() outright; skipping
    it here (exactly like the "nothing to apply yet" cases above) avoids the
    crash and its knock-on effect of desyncing other widgets' displayed
    values by forcing an unplanned extra rerun.

    The FIRST correctly-shaped snapshot of a session is *also* discarded,
    not folded in -- see CANVAS_WARMED_UP. This is a defensive measure for a
    bug that only shows up on Streamlit Community Cloud, not locally (see
    _patch_image_to_url_compat's own docstring for the general background-
    loading race this component already has, and which this addresses a
    variant of): the background image can apparently still be mid-load when
    the browser reports back this first snapshot, so what comes back isn't
    the intended all-(-1000 HU) phantom but whatever the page's own theme
    background looked like underneath -- and unlike the shape-mismatch case,
    this snapshot has the *right* shape, so nothing before this would have
    caught it. Folding it in would silently overwrite the correctly
    initialized blank_image_hu() with that wrong color, which then has to be
    manually re-painted over. Since a real user stroke can't physically
    happen before the canvas has rendered at least once, discarding
    specifically the first snapshot -- and only that one -- never discards
    real input; at worst, if the canvas is still mid-load *and* the user's
    first stroke both land in the same rerun, the very start of that one
    stroke is dropped (the rest of it, and everything after, is folded in
    normally, since `realtime_update=True` reports a stroke-in-progress
    across several reruns, not just one at the end).
    """
    if canvas_result is None or canvas_result.image_data is None:
        return
    expected_shape = (config.IMAGE_SIZE_PX, config.IMAGE_SIZE_PX)
    if canvas_result.image_data.shape[:2] != expected_shape:
        return
    if not st.session_state[state.CANVAS_WARMED_UP]:
        st.session_state[state.CANVAS_WARMED_UP] = True
        return
    gray = canvas_result.image_data[:, :, :3].mean(axis=2)
    hu = gray_u8_to_hu(gray)
    st.session_state[state.IMAGE_HU] = ct_model.enforce_constraints(hu)
