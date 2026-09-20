# Parallel-Beam CT Simulator

An interactive teaching tool for x-ray computed tomography: draw a grayscale
phantom, manually rotate a virtual detector with an angle slider to see its
live line integral p(t), and run a simulated CT scan to acquire the full
sinogram.

Implements the model in `Specification.txt`: a 500x500 pixel circular
phantom (50 cm field of view, -1000 to +1000 HU), a 500x900 sinogram
(t = -24.95..24.95 cm, theta = 0..179.8 deg), and a 5-second animated scan.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

1. Push this directory to a GitHub repo (`app.py`, `requirements.txt`, and
   the other `.py` modules all need to be in the repo root, or adjust the
   app's main file path in the Community Cloud dashboard accordingly).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at that repo/branch, with **Main file path** set to `app.py`.
3. No secrets or extra configuration are needed -- everything the app uses
   is self-contained in `requirements.txt`.

Community Cloud installs from `requirements.txt` automatically; there's
nothing else to configure.

## Project structure

| File | Role |
|---|---|
| `config.py` | Every constant from the spec (geometry, HU/mu conversion, sinogram grid, scan timing), plus a few UI defaults (brush sizes, canvas margin) -- the single place to retune any of them. |
| `state.py` | `st.session_state` schema and defaults. |
| `ct_model.py` | Pure physics: HU<->mu, the phantom circle mask, single-angle and full-sinogram projections (via `skimage.transform.radon`). No Streamlit imports. |
| `plotting.py` | Pure matplotlib rendering (image/arrows panel, p(t) plot, sinogram, HU colorbar). No Streamlit imports. |
| `drawing.py` | The paint canvas (`streamlit-drawable-canvas`) plus the HU/brush-size controls. Includes a compatibility shim for a Streamlit-internals change the canvas package hasn't caught up to (see comment in the file). |
| `scan.py` | The "Scan" button: computes the full sinogram once, then animates a time-adaptive left-to-right reveal that self-paces to `T_SCAN_SECONDS` regardless of the host machine's rendering speed. |
| `app.py` | Layout and wiring only -- the sole place that decides what's rendered where. |

## Notes

- **Total time from clicking Scan to a fully settled screen is longer than
  the animated 5-second sweep itself** -- computing the full sinogram
  up front (before the sweep starts) takes a variable amount of time
  depending on the host machine's load. The animated sweep is what's paced
  to 5 seconds; the one-time computation before it is not.
- The interactive canvas requires a real mouse/touch drag; it isn't
  keyboard-accessible. Gantry rotation, however, is a native slider and is
  keyboard-accessible.
- The 10 cm reference grid is drawn only in the gantry sketch panel (the
  read-only reproduction of the phantom), not in the paint canvas itself --
  the canvas's full raster is folded back into `image_hu` on every rerun, so
  any grid baked into its background would permanently contaminate the
  drawn phantom.
