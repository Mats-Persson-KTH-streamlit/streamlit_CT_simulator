"""Canvas-based painting: HU/brush-size controls plus the drawing canvas."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.elements.image as _st_image_module
from PIL import Image

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


def hu_to_gray_u8(hu_array):
    frac = (hu_array - config.HU_MIN) / (config.HU_MAX - config.HU_MIN)
    return np.clip(frac * 255, 0, 255).astype(np.uint8)


def gray_u8_to_hu(gray_array):
    frac = gray_array.astype(np.float64) / 255.0
    return config.HU_MIN + frac * (config.HU_MAX - config.HU_MIN)


def hu_to_gray_hex(hu_value):
    g = int(np.clip((hu_value - config.HU_MIN) / (config.HU_MAX - config.HU_MIN) * 255, 0, 255))
    return f"#{g:02x}{g:02x}{g:02x}"


def image_hu_to_pil(image_hu):
    """Phantom rendered at 1:1 scale as the canvas background."""
    gray = hu_to_gray_u8(image_hu)
    return Image.fromarray(gray, mode="L").convert("RGB")


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
    """Renders the paint surface and returns the raw CanvasResult."""
    background = image_hu_to_pil(st.session_state[state.IMAGE_HU])
    stroke_color = hu_to_gray_hex(st.session_state[state.BRUSH_HU])
    stroke_width = 2 * st.session_state[state.BRUSH_SIZE_PX]

    return st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=background,
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
