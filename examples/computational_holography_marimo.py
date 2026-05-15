import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Computational Holography with `slmsuite`

    This marimo notebook is a reactive version of the
    [`computational_holography.ipynb`](https://slmsuite.readthedocs.io/en/latest/_examples/computational_holography.html)
    tutorial used in the `slmsuite` documentation.

    It follows the same progression:

    - connect the SLM nearfield to the farfield by the Fourier transform
    - make a single-pixel farfield target and solve the nearfield phase with Gerchberg-Saxton
    - inspect how padding and illumination shape change the reconstruction
    - build spot arrays with `SpotHologram`
    - compare GS against fixed-phase weighted GS
    - form an image from the `slmsuite` logo

    `slmsuite` uses CuPy automatically when it is installed; otherwise the same code falls back to NumPy.
    """)
    return


@app.cell
def _():
    import copy
    import os
    import sys
    import warnings
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.image import imread

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    mpl.rc("image", cmap="inferno")
    os.environ.setdefault("MPLBACKEND", "Agg")
    return copy, imread, np, plt, repo_root, warnings


@app.cell
def _(warnings):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="cupy is not installed; using numpy.*",
        )
        from slmsuite.hardware.cameras.simulated import SimulatedCamera
        from slmsuite.hardware.cameraslms import FourierSLM
        from slmsuite.hardware.slms.simulated import SimulatedSLM
        from slmsuite.holography import toolbox
        from slmsuite.holography.algorithms import Hologram, SpotHologram

    return (
        FourierSLM,
        Hologram,
        SimulatedCamera,
        SimulatedSLM,
        SpotHologram,
        toolbox,
    )


@app.cell
def _(mo, np, plt):
    def collect_figures(plotter):
        """Run a plotting callback and return its result plus newly created figures."""
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        plt.close("all")
        try:
            result = plotter()
            current = set(plt.get_fignums())
        finally:
            plt.show = original_show

        figures = [plt.figure(number) for number in sorted(current)]
        return result, figures

    def render_figures(figures, columns=1):
        if not figures:
            return mo.md("_No figures were generated._")

        rendered = [mo.as_html(figure) for figure in figures]
        for figure in figures:
            plt.close(figure)

        if columns <= 1:
            return mo.vstack(rendered, gap=1.0)

        rows = [
            mo.hstack(rendered[index : index + columns], widths="equal", gap=1.0)
            for index in range(0, len(rendered), columns)
        ]
        return mo.vstack(rows, gap=1.0)

    def make_single_pixel_target(row, col, shape=(32, 32)):
        target = np.zeros(shape)
        target[int(row), int(col)] = 1
        return target

    def stats_summary(stats_dict, group):
        stats = stats_dict["stats"][group]
        efficiency = stats["efficiency"][-1]
        uniformity = stats["uniformity"][-1]
        pkpk_err = stats["pkpk_err"][-1]
        std_err = stats["std_err"][-1]
        return {
            "efficiency": float(efficiency),
            "uniformity": float(uniformity),
            "pkpk_err": float(pkpk_err),
            "std_err": float(std_err),
        }

    return (
        collect_figures,
        make_single_pixel_target,
        render_figures,
        stats_summary,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## SLM Fundamentals

    A phase-mode spatial light modulator delays the optical wavefront pixel by pixel.
    A linear phase ramp across the display acts like a tilted mirror: it steers light
    to a nonzero farfield angle. Because practical SLMs only cover about \(2\pi\) of phase,
    the ramp is wrapped like a Fresnel lens.

    The SLM plane and farfield plane are Fourier pairs. A spatial frequency in the SLM
    nearfield becomes a spot position in the farfield, often described in normalized
    \(k\)-space:

    \[
    k_x = \frac{\vec{k} \cdot \hat{x}}{|\vec{k}|}.
    \]

    For the small angles common in SLM systems, this normalized coordinate is approximately
    the steering angle \(\theta\). A lens can place the Fourier plane at an accessible image
    plane, making the camera image a practical view of the computed farfield.

    ![Beam steering in one and two dimensions](https://raw.githubusercontent.com/holodyne/slmsuite-examples/main/examples/fourier_shearing_mod2_small.gif)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Beam Steering in `slmsuite`

    Start with the smallest useful target: a single illuminated pixel in a \(32 \times 32\)
    farfield grid. In `slmsuite`, the target and SLM arrays are Fourier pairs, so each target
    pixel corresponds to a spatial frequency in the SLM plane. The controls below let you move
    that target pixel and change the number of GS iterations used to recover a phase mask.
    """)
    return


@app.cell
def _(mo):
    single_pixel_controls = mo.md(
        """
        **Single-pixel + GS controls**

        These controls drive the target, coordinate-system plots, GS phase retrieval,
        and the diffraction-resolution section below.

        Row: {row}

        Column: {col}

        GS iterations: {iterations}
        """
    ).batch(
        row=mo.ui.slider(0, 31, step=1, value=9, show_value=True),
        col=mo.ui.slider(0, 31, step=1, value=24, show_value=True),
        iterations=mo.ui.slider(1, 15, step=1, value=5, show_value=True),
    )
    single_pixel_run = mo.ui.run_button(label="Recompute single-pixel figures")

    return single_pixel_controls, single_pixel_run


@app.cell
def _(single_pixel_controls):
    single_pixel_col = int(single_pixel_controls.value["col"])
    single_pixel_iterations = int(single_pixel_controls.value["iterations"])
    single_pixel_row = int(single_pixel_controls.value["row"])
    return single_pixel_col, single_pixel_iterations, single_pixel_row


@app.cell
def _(
    FourierSLM,
    Hologram,
    SimulatedCamera,
    SimulatedSLM,
    collect_figures,
    make_single_pixel_target,
    single_pixel_col,
    single_pixel_iterations,
    single_pixel_run,
    single_pixel_row,
    warnings,
):
    single_pixel_run.value

    single_target_shape = (32, 32)
    single_target = make_single_pixel_target(
        single_pixel_row,
        single_pixel_col,
        shape=single_target_shape,
    )
    single_hologram = Hologram(single_target, slm_shape=single_target_shape)

    single_wav_um = 0.532
    single_slm = SimulatedSLM(
        (single_target_shape[1], single_target_shape[0]),
        pitch_um=(10, 10),
        wav_um=single_wav_um,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"'camera' _get_image_hw\(\) failed .*",
        )
        single_camera = SimulatedCamera(single_slm)
    single_setup = FourierSLM(single_camera, single_slm)
    single_hologram.cameraslm = single_setup

    single_zoombox, single_target_figures = collect_figures(
        lambda: single_hologram.plot_farfield(
            source=single_hologram.target,
            cbar=True,
            title="Single-pixel target",
        )
    )

    _, single_units_figures = collect_figures(
        lambda: [
            single_hologram.plot_farfield(
                source=single_hologram.target,
                units=units,
                figsize=(6, 3),
                title=f"Target in {units}",
            )
            for units in ["kxy", "freq", "deg"]
        ]
    )

    single_gs_hologram = Hologram(single_target, slm_shape=single_target_shape)
    single_gs_hologram.cameraslm = single_setup
    single_gs_hologram.optimize(
        method="GS",
        maxiter=single_pixel_iterations,
        verbose=False,
    )

    _, single_gs_figures = collect_figures(
        lambda: (
            single_gs_hologram.plot_nearfield(
                title=f"GS after {single_pixel_iterations} iterations",
                cbar=True,
            ),
            single_gs_hologram.plot_farfield(
                limits=single_zoombox,
                cbar=True,
                title="GS farfield amplitude",
            ),
        )
    )
    return (
        single_gs_figures,
        single_setup,
        single_slm,
        single_target,
        single_target_figures,
        single_units_figures,
        single_zoombox,
    )


@app.cell
def _(
    mo,
    render_figures,
    single_gs_figures,
    single_pixel_controls,
    single_pixel_run,
    single_target_figures,
    single_units_figures,
):
    mo.vstack(
        [
            mo.vstack([single_pixel_controls, single_pixel_run], gap=1.0),
            mo.md(
                r"""
                ### Single-pixel target and coordinate systems

                The target appears in the computational `"knm"` basis first. With simulated SLM
                and camera objects attached, `plot_farfield` can re-label the same target in
                normalized \(k\)-space, spatial frequency, and degrees.
                """
            ),
            render_figures(single_target_figures),
            render_figures(single_units_figures),
            mo.md(
                """
                ### GS phase retrieval

                Gerchberg-Saxton alternates between the nearfield and farfield, enforcing the
                known nearfield amplitude and the requested farfield amplitude while solving
                for the unknown nearfield phase.
                """
            ),
            render_figures(single_gs_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Resolving the diffraction-limited spot

    The single-pixel target can look deceptively perfect because the farfield grid is sampled
    at about the diffraction-limited spot size. Zero-padding the nearfield increases the
    computational farfield resolution, so the actual spot shape becomes visible. A Gaussian
    source amplitude similarly broadens the farfield because the illuminated aperture is
    effectively smaller.
    """)
    return


@app.cell
def _(mo):
    diffraction_controls = mo.md(
        """
        **Diffraction-resolution controls**

        Farfield sample width / diffraction-limited width: {farfield_fraction}

        Gaussian source width in fractional SLM coordinates: {gaussian_width}
        """
    ).batch(
        farfield_fraction=mo.ui.slider(
            0.25,
            1.0,
            step=0.25,
            value=0.5,
            show_value=True,
        ),
        gaussian_width=mo.ui.slider(
            0.15,
            0.5,
            step=0.05,
            value=0.25,
            show_value=True,
        ),
    )
    diffraction_run = mo.ui.run_button(label="Recompute diffraction figures")

    return diffraction_controls, diffraction_run


@app.cell
def _(diffraction_controls):
    farfield_fraction = float(diffraction_controls.value["farfield_fraction"])
    gaussian_width = float(diffraction_controls.value["gaussian_width"])
    return farfield_fraction, gaussian_width


@app.cell
def _(
    Hologram,
    SimulatedSLM,
    collect_figures,
    diffraction_run,
    farfield_fraction,
    gaussian_width,
    single_pixel_iterations,
    single_pixel_run,
    single_setup,
    single_slm,
    single_target,
    single_zoombox,
    toolbox,
):
    single_pixel_run.value
    diffraction_run.value

    padding_probe = Hologram(single_target, slm_shape=single_target.shape)
    padded_shape = padding_probe.get_padded_shape(
        single_setup,
        precision=farfield_fraction / (single_slm.pitch[0] * single_slm.shape[0]),
    )
    padded_target = toolbox.pad(single_target, padded_shape)
    padded_hologram = Hologram(padded_target, slm_shape=single_slm.shape)
    padded_hologram.cameraslm = single_setup

    padded_zoombox, padded_target_figures = collect_figures(
        lambda: padded_hologram.plot_farfield(
            padded_target,
            title="Zero-padded target",
        )
    )

    padded_hologram.optimize(
        method="GS",
        maxiter=single_pixel_iterations,
        verbose=False,
    )

    _, padded_gs_figures = collect_figures(
        lambda: (
            padded_hologram.plot_nearfield(
                padded=True,
                cbar=True,
                title="Zero-padded GS nearfield",
            ),
            padded_hologram.plot_farfield(
                cbar=True,
                units="deg",
                title="Zero-padded GS farfield",
                limits=padded_zoombox,
            ),
        )
    )

    gaussian_slm = SimulatedSLM(
        (single_target.shape[1], single_target.shape[0]),
        pitch_um=(10, 10),
        wav_um=0.532,
    )
    gaussian_slm.set_source_analytic(
        units="frac",
        x0=0,
        y0=0,
        a=1,
        c=0,
        wx=gaussian_width,
        wy=gaussian_width,
    )
    gaussian_hologram = Hologram(
        single_target,
        slm_shape=gaussian_slm.shape,
        amp=gaussian_slm.source["amplitude"],
    )
    gaussian_hologram.optimize(
        method="GS",
        maxiter=single_pixel_iterations,
        verbose=False,
    )

    _, gaussian_figures = collect_figures(
        lambda: (
            gaussian_hologram.plot_nearfield(
                padded=True,
                cbar=True,
                title="Gaussian source",
            ),
            gaussian_hologram.plot_farfield(
                cbar=True,
                limits=single_zoombox,
                title="Gaussian-source farfield",
            ),
        )
    )
    return (
        gaussian_figures,
        padded_gs_figures,
        padded_shape,
        padded_target_figures,
    )


@app.cell
def _(
    gaussian_figures,
    diffraction_controls,
    diffraction_run,
    mo,
    padded_gs_figures,
    padded_shape,
    padded_target_figures,
    render_figures,
):
    mo.vstack(
        [
            mo.vstack([diffraction_controls, diffraction_run], gap=1.0),
            mo.md(f"Zero-padding expands the computational grid to `{tuple(padded_shape)}`."),
            render_figures(padded_target_figures),
            render_figures(padded_gs_figures),
            mo.md(
                "With Gaussian illumination, less of the SLM aperture is effectively used, "
                "so the farfield feature is wider."
            ),
            render_figures(gaussian_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spot Arrays

    `SpotHologram` specializes `Hologram` for arrays of optical foci. Spot arrays are central
    to optical tweezers, atom-array experiments, optogenetics, and other applications where
    power must be distributed across many isolated target sites.

    The next section is heavier than the single-pixel demo, so it is gated by a run button.
    Keep the SLM side length at `512` for a quick CPU preview, or use `1024` to match the
    original documentation example more closely.
    """)
    return


@app.cell
def _(mo):
    spot_controls = mo.md(
        """
        **Spot-array controls**

        SLM side length: {slm_side}

        Array side length: {array_side}

        Array pitch in `"knm"` pixels: {array_pitch}

        GS/WGS iterations: {iterations}
        """
    ).batch(
        slm_side=mo.ui.slider(256, 1024, step=256, value=512, show_value=True),
        array_side=mo.ui.slider(4, 12, step=1, value=10, show_value=True),
        array_pitch=mo.ui.slider(6, 16, step=1, value=10, show_value=True),
        iterations=mo.ui.slider(5, 31, step=2, value=21, show_value=True),
    )
    spot_run = mo.ui.run_button(label="Run spot-array GS/WGS")

    mo.vstack([spot_controls, spot_run], gap=1.0)
    return spot_controls, spot_run


@app.cell
def _(spot_controls):
    spot_array_pitch = int(spot_controls.value["array_pitch"])
    spot_array_side = int(spot_controls.value["array_side"])
    spot_iterations = int(spot_controls.value["iterations"])
    spot_slm_side = int(spot_controls.value["slm_side"])
    return spot_array_pitch, spot_array_side, spot_iterations, spot_slm_side


@app.cell
def _(
    SpotHologram,
    collect_figures,
    mo,
    render_figures,
    spot_array_pitch,
    spot_array_side,
    spot_iterations,
    spot_run,
    spot_slm_side,
    stats_summary,
):
    mo.stop(
        not spot_run.value,
        mo.md("Click **Run spot-array GS/WGS** to compute this section."),
    )

    spot_hologram = SpotHologram.make_rectangular_array(
        (spot_slm_side, spot_slm_side),
        array_shape=(spot_array_side, spot_array_side),
        array_pitch=(spot_array_pitch, spot_array_pitch),
        basis="knm",
    )
    spot_zoom, spot_target_figures = collect_figures(
        lambda: spot_hologram.plot_farfield(
            source=spot_hologram.target,
            title="Target farfield",
        )
    )

    spot_callback_figures = []

    def plot_spot_callback(hologram):
        if hologram.iter % 10 == 0:
            _, figures = collect_figures(
                lambda: hologram.plot_farfield(
                    limits=spot_zoom,
                    title=f"GS iteration {hologram.iter}",
                )
            )
            spot_callback_figures.extend(figures)

    spot_hologram.optimize(
        method="GS",
        maxiter=spot_iterations,
        callback=plot_spot_callback,
        stat_groups=["computational_spot"],
        verbose=False,
    )
    spot_gs_summary = stats_summary(spot_hologram.stats, "computational_spot")

    _, spot_summary_figures = collect_figures(
        lambda: (
            spot_hologram.plot_stats(),
            spot_hologram.plot_nearfield(title="Final GS nearfield"),
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ### GS spot-array result

                Final computational spot metrics:

                - efficiency: `{spot_gs_summary["efficiency"]:.3f}`
                - uniformity: `{spot_gs_summary["uniformity"]:.3f}`
                - peak-to-peak error: `{spot_gs_summary["pkpk_err"]:.3f}`
                - standard-deviation error: `{spot_gs_summary["std_err"]:.3f}`
                """
            ),
            render_figures(spot_target_figures),
            render_figures(spot_callback_figures, columns=2),
            render_figures(spot_summary_figures),
        ],
        gap=1.25,
    )
    return spot_hologram, spot_zoom


@app.cell
def _(
    collect_figures,
    copy,
    mo,
    render_figures,
    spot_hologram,
    spot_iterations,
    spot_zoom,
    stats_summary,
    warnings,
):
    wgs_hologram = copy.deepcopy(spot_hologram)
    wgs_hologram.reset()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in divide",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in divide",
        )
        wgs_hologram.optimize(
            method="WGS-Kim",
            maxiter=spot_iterations,
            verbose=False,
            stat_groups=["computational_spot"],
        )
    wgs_summary = stats_summary(wgs_hologram.stats, "computational_spot")

    _, wgs_farfield_figures = collect_figures(
        lambda: wgs_hologram.plot_farfield(
            limits=spot_zoom,
            title="WGS-Kim farfield",
        )
    )

    combined_stats = copy.deepcopy(wgs_hologram.stats)
    combined_stats["stats"]["WGS"] = combined_stats["stats"].pop("computational_spot")
    combined_stats["stats"]["GS"] = spot_hologram.stats["stats"]["computational_spot"]
    combined_stats["stats"] = dict(reversed(list(combined_stats["stats"].items())))

    _, wgs_comparison_figures = collect_figures(
        lambda: wgs_hologram.plot_stats(combined_stats)
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ### Fixed-phase WGS comparison

                `WGS-Kim` fixes the farfield phase after its configured phase-fixing condition,
                then keeps reweighting target amplitudes to improve spot uniformity.

                Final WGS computational spot metrics:

                - efficiency: `{wgs_summary["efficiency"]:.3f}`
                - uniformity: `{wgs_summary["uniformity"]:.3f}`
                - peak-to-peak error: `{wgs_summary["pkpk_err"]:.3f}`
                - standard-deviation error: `{wgs_summary["std_err"]:.3f}`

                Method flags: `{wgs_hologram.flags}`
                """
            ),
            render_figures(wgs_farfield_figures),
            render_figures(wgs_comparison_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Basic Image Formation

    The same phase-retrieval machinery can target dense images, not just isolated spots.
    A perfect result is not generally possible when the nearfield amplitude is fixed, but
    WGS can still produce a close reconstruction. This section uses the `slmsuite` logo from
    the repository docs.
    """)
    return


@app.cell
def _(mo):
    image_controls = mo.md(
        """
        **Image-formation controls**

        Computational shape: {shape_side}

        WGS iterations: {iterations}
        """
    ).batch(
        shape_side=mo.ui.slider(1024, 2048, step=512, value=1024, show_value=True),
        iterations=mo.ui.slider(5, 31, step=2, value=21, show_value=True),
    )
    image_run = mo.ui.run_button(label="Run image formation")

    mo.vstack([image_controls, image_run], gap=1.0)
    return image_controls, image_run


@app.cell
def _(image_controls):
    image_iterations = int(image_controls.value["iterations"])
    image_shape_side = int(image_controls.value["shape_side"])
    return image_iterations, image_shape_side


@app.cell
def _(
    Hologram,
    collect_figures,
    image_iterations,
    image_run,
    image_shape_side,
    imread,
    mo,
    np,
    render_figures,
    repo_root,
    toolbox,
    warnings,
):
    mo.stop(
        not image_run.value,
        mo.md("Click **Run image formation** to compute this section."),
    )

    logo_path = repo_root / "docs" / "source" / "static" / "slmsuite-small.png"
    logo_image = imread(logo_path)
    if logo_image.ndim == 3:
        logo_gray = logo_image[..., :3].mean(axis=2)
    else:
        logo_gray = logo_image
    logo_target = toolbox.pad(1 - logo_gray, (image_shape_side, image_shape_side))

    image_hologram = Hologram(logo_target)
    image_zoom, image_target_figures = collect_figures(
        lambda: image_hologram.plot_farfield(
            image_hologram.target,
            cbar=True,
            title="Target farfield",
        )
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in divide",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in divide",
        )
        image_hologram.optimize(
            method="WGS-Kim",
            maxiter=image_iterations,
            verbose=False,
        )
    _, image_result_figures = collect_figures(
        lambda: (
            image_hologram.plot_farfield(
                limits=image_zoom,
                title="WGS image reconstruction",
                cbar=True,
            ),
            image_hologram.plot_farfield(
                np.abs(image_hologram.target - image_hologram.amp_ff),
                limits=image_zoom,
                title="WGS image error",
                cbar=True,
            ),
        )
    )

    mo.vstack(
        [
            render_figures(image_target_figures),
            render_figures(image_result_figures),
            mo.md(
                """
                The error plot highlights the phase-only constraint: the SLM can choose phase,
                but the source amplitude remains fixed, so the requested amplitude image can only
                be approximated.
                """
            ),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## What to take away

    - `Hologram` solves a Fourier phase-retrieval problem: fixed nearfield amplitude,
      desired farfield amplitude, unknown nearfield phase.
    - `SimulatedSLM`, `SimulatedCamera`, and `FourierSLM` provide the geometry needed to
      express farfield positions in physical units.
    - Zero-padding changes computational sampling, not the physical aperture.
    - Source amplitude matters: narrower illumination creates wider diffraction features.
    - `SpotHologram` adds target construction and spot-specific statistics for focus arrays.
    - Weighted GS methods trade some efficiency for much better uniformity.
    """)
    return


if __name__ == "__main__":
    app.run()
