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
    # Multipoint Wavefront Calibration with `slmsuite`

    This marimo notebook adapts the multipoint calibration documentation example to a
    compact simulated SLM/camera pair. It demonstrates how calibration points are chosen,
    how simultaneous superpixel wavefront tests are run, and how multipoint Zernike
    measurements can be smoothed before reuse.
    """)
    return


@app.cell
def _():
    import os
    import sys
    import warnings
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    mpl.rc("image", cmap="inferno")
    os.environ.setdefault("MPLBACKEND", "Agg")
    return np, plt, repo_root, warnings


@app.cell
def _(warnings):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cupy is not installed; using numpy.*")
        from slmsuite.hardware.cameras.simulated import SimulatedCamera
        from slmsuite.hardware.cameraslms import FourierSLM
        from slmsuite.hardware.slms.simulated import SimulatedSLM
        from slmsuite.holography import toolbox
        from slmsuite.holography.toolbox import phase as phase_tools

    return FourierSLM, SimulatedCamera, SimulatedSLM, phase_tools, toolbox


@app.cell
def _(mo, np, plt):
    def collect_figures(plotter):
        """Run a plotting callback and return its result plus matplotlib figures."""
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        plt.close("all")
        try:
            result = plotter()
            figures = [plt.figure(number) for number in sorted(plt.get_fignums())]
        finally:
            plt.show = original_show
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

    def render_frame_strip(frames, max_frames=5):
        if frames is None or len(frames) == 0:
            return mo.md("_No movie frames were generated._")

        frame_indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)

        def _plot_frames():
            fig, axs = plt.subplots(1, len(frame_indices), figsize=(3.6 * len(frame_indices), 3))
            if len(frame_indices) == 1:
                axs = [axs]
            for ax, frame_index in zip(axs, frame_indices):
                ax.imshow(frames[frame_index])
                ax.set_title(f"frame {frame_index + 1}")
                ax.axis("off")
            fig.tight_layout()

        _, figures = collect_figures(_plot_frames)
        return render_figures(figures)

    def corner_points(camera_shape, x_fraction=0.24, y_fraction=0.28):
        height, width = camera_shape
        xs = np.array([x_fraction * width, (1 - x_fraction) * width])
        ys = np.array([y_fraction * height, (1 - y_fraction) * height])
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.vstack((grid_x.ravel(), grid_y.ravel()))

    def plot_point_map(camera_shape, generated_points, scheduled_points):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(np.zeros(camera_shape), cmap="gray", vmin=0, vmax=1)
        if generated_points is not None and generated_points.size > 0:
            ax.scatter(
                generated_points[0],
                generated_points[1],
                s=52,
                facecolors="none",
                edgecolors="tab:cyan",
                linewidth=1.6,
                label="wavefront_calibration_points",
            )
        ax.scatter(
            scheduled_points[0],
            scheduled_points[1],
            s=70,
            marker="x",
            color="tab:red",
            linewidth=2.0,
            label="scheduled compact set",
        )
        ax.set_xlim(0, camera_shape[1] - 1)
        ax.set_ylim(camera_shape[0] - 1, 0)
        ax.set_xlabel("Camera i [pix]")
        ax.set_ylabel("Camera j [pix]")
        ax.set_title("Multipoint calibration locations")
        ax.legend(loc="upper right")
        return ax

    return (
        collect_figures,
        corner_points,
        plot_point_map,
        render_figures,
        render_frame_strip,
    )


@app.cell
def _(mo):
    setup_controls = mo.md(
        """
        **Multipoint simulation setup**

        SLM width: {slm_width}

        Camera width: {camera_width}

        Aberration scale: {aberration_scale}

        Camera gain: {gain}
        """
    ).batch(
        slm_width=mo.ui.slider(240, 480, step=80, value=320, show_value=True),
        camera_width=mo.ui.slider(220, 440, step=68, value=288, show_value=True),
        aberration_scale=mo.ui.slider(0.0, 2.0, step=0.25, value=1.0, show_value=True),
        gain=mo.ui.slider(50, 300, step=25, value=120, show_value=True),
    )
    setup_run = mo.ui.run_button(label="Reinitialize multipoint simulation")

    mo.vstack([setup_controls, setup_run], gap=1.0)
    return setup_controls, setup_run


@app.cell
def _(setup_controls):
    mp_aberration_scale = float(setup_controls.value["aberration_scale"])
    mp_camera_width = int(setup_controls.value["camera_width"])
    mp_gain = float(setup_controls.value["gain"])
    mp_slm_width = int(setup_controls.value["slm_width"])
    return mp_aberration_scale, mp_camera_width, mp_gain, mp_slm_width


@app.cell
def _(
    FourierSLM,
    SimulatedCamera,
    SimulatedSLM,
    collect_figures,
    mp_aberration_scale,
    mp_camera_width,
    mp_gain,
    mp_slm_width,
    np,
    phase_tools,
    setup_run,
    warnings,
):
    setup_run.value

    mp_slm_resolution = (mp_slm_width, int(round(mp_slm_width * 0.75)))
    mp_camera_resolution = (mp_camera_width, int(round(mp_camera_width * 0.76)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"'camera' _get_image_hw\(\) failed .*")
        mp_slm = SimulatedSLM(mp_slm_resolution, pitch_um=(8, 8))
        mp_phase_aberration = mp_aberration_scale * phase_tools.zernike_sum(
            mp_slm,
            indices=(3, 4, 5, 7, 8),
            weights=(1, -2, 2, 0.8, -0.6),
            aperture=None,
            use_mask=False,
        )
        mp_slm.set_source_analytic(phase_offset=mp_phase_aberration, sim=True)
        mp_slm.set_source_analytic()
        mp_cam = SimulatedCamera(
            mp_slm,
            mp_camera_resolution,
            pitch_um=(4, 4),
            gain=mp_gain,
        )

    mp_fs = FourierSLM(mp_cam, mp_slm)
    mp_M, mp_b = mp_fs.fourier_calibration_build(
        f_eff=80000.0,
        theta=5 * np.pi / 180,
    )
    mp_cam.set_affine(mp_M, mp_b)
    mp_fs.fourier_calibrate_analytic(mp_M, mp_b)
    mp_cam.set_exposure(0.02)

    _, mp_source_figures = collect_figures(lambda: mp_slm.plot_source(sim=True))
    return mp_M, mp_b, mp_cam, mp_fs, mp_slm, mp_source_figures


@app.cell
def _(mo, np, render_figures, mp_M, mp_b, mp_cam, mp_slm, mp_source_figures):
    mo.vstack(
        [
            mo.md(
                f"""
                Active simulated system:

                - SLM shape: `{mp_slm.shape}` pixels
                - Camera shape: `{mp_cam.shape}` pixels
                - Fourier matrix determinant: `{np.linalg.det(mp_M):.3g}`
                - Fourier offset: `{np.array2string(np.ravel(mp_b), precision=3)}`
                """
            ),
            render_figures(mp_source_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Calibration Points

    `wavefront_calibration_points` proposes points in camera coordinates. The compact
    scheduled set used by the heavier cells below is deliberately sparse so the simulated
    interference windows do not overlap.
    """)
    return


@app.cell
def _(mo):
    point_controls = mo.md(
        """
        **Point selection controls**

        Generated point pitch: {point_pitch}

        Compact x inset fraction: {x_inset}

        Compact y inset fraction: {y_inset}
        """
    ).batch(
        point_pitch=mo.ui.slider(60, 220, step=20, value=140, show_value=True),
        x_inset=mo.ui.slider(0.18, 0.34, step=0.02, value=0.24, show_value=True),
        y_inset=mo.ui.slider(0.20, 0.36, step=0.02, value=0.28, show_value=True),
    )
    point_run = mo.ui.run_button(label="Regenerate calibration points")

    mo.vstack([point_controls, point_run], gap=1.0)
    return point_controls, point_run


@app.cell
def _(point_controls):
    point_pitch = float(point_controls.value["point_pitch"])
    x_inset = float(point_controls.value["x_inset"])
    y_inset = float(point_controls.value["y_inset"])
    return point_pitch, x_inset, y_inset


@app.cell
def _(
    collect_figures,
    corner_points,
    mp_cam,
    mp_fs,
    plot_point_map,
    point_pitch,
    point_run,
    x_inset,
    y_inset,
):
    point_run.value

    generated_points_ij, generated_point_figures = collect_figures(
        lambda: mp_fs.wavefront_calibration_points(
            pitch=point_pitch,
            avoid_mirrors=True,
            plot=True,
        )
    )
    scheduled_points_ij = corner_points(mp_cam.shape, x_fraction=x_inset, y_fraction=y_inset)
    _, scheduled_point_figures = collect_figures(
        lambda: plot_point_map(mp_cam.shape, generated_points_ij, scheduled_points_ij)
    )
    return generated_point_figures, generated_points_ij, scheduled_point_figures, scheduled_points_ij


@app.cell
def _(
    generated_point_figures,
    generated_points_ij,
    mo,
    render_figures,
    scheduled_point_figures,
    scheduled_points_ij,
):
    mo.vstack(
        [
            mo.md(
                f"""
                Generated points: `{generated_points_ij.shape[1]}`.
                Scheduled compact points: `{scheduled_points_ij.shape[1]}`.
                """
            ),
            render_figures(generated_point_figures),
            render_figures(scheduled_point_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Simultaneous Superpixel Test

    The original notebook uses `plot=3` to build a phase-sweep movie. Here the returned
    frames are shown as a strip, and the same compact point set is used at all scheduled
    calibration locations.
    """)
    return


@app.cell
def _(mo):
    test_controls = mo.md(
        """
        **Superpixel test controls**

        Field point in frequency units: {field_freq}

        Superpixel size: {superpixel_size}

        Test index: {test_index}

        Phase steps: {phase_steps}
        """
    ).batch(
        field_freq=mo.ui.slider(0.05, 0.25, step=0.01, value=0.15, show_value=True),
        superpixel_size=mo.ui.slider(80, 160, step=20, value=120, show_value=True),
        test_index=mo.ui.slider(0, 8, step=1, value=2, show_value=True),
        phase_steps=mo.ui.slider(3, 8, step=1, value=4, show_value=True),
    )
    test_run = mo.ui.run_button(label="Run simultaneous superpixel test")

    mo.vstack([test_controls, test_run], gap=1.0)
    return test_controls, test_run


@app.cell
def _(test_controls):
    test_field_freq = float(test_controls.value["field_freq"])
    test_index = int(test_controls.value["test_index"])
    test_phase_steps = int(test_controls.value["phase_steps"])
    test_superpixel_size = int(test_controls.value["superpixel_size"])
    return test_field_freq, test_index, test_phase_steps, test_superpixel_size


@app.cell
def _(
    mo,
    mp_fs,
    np,
    render_frame_strip,
    scheduled_points_ij,
    test_field_freq,
    test_index,
    test_phase_steps,
    test_run,
    test_superpixel_size,
    warnings,
):
    mo.stop(
        not test_run.value,
        mo.md("Click **Run simultaneous superpixel test** to compute this section."),
    )

    _active_shape = mp_fs.slm.shape
    _supershape = (
        int(np.ceil(_active_shape[0] / test_superpixel_size)),
        int(np.ceil(_active_shape[1] / test_superpixel_size)),
    )
    _safe_test_index = min(test_index, _supershape[0] * _supershape[1] - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_movie = mp_fs.wavefront_calibrate_superpixel(
            calibration_points=scheduled_points_ij,
            field_point=(test_field_freq, 0),
            field_point_units="freq",
            superpixel_size=test_superpixel_size,
            test_index=_safe_test_index,
            phase_steps=test_phase_steps,
            corrected_amplitude=True,
            plot=3,
        )

    mo.vstack(
        [
            mo.md(
                f"Returned `{len(test_movie)}` phase-sweep frames for scheduled index `{_safe_test_index}`."
            ),
            render_frame_strip(test_movie),
        ],
        gap=1.0,
    )
    return test_movie


@app.cell
def _(mo):
    mo.md(r"""
    ## Compact Full Superpixel Calibration

    This runs a small multipoint superpixel calibration and then processes two point-specific
    corrections. The settings are intentionally compact; increasing the number of phase
    steps or reducing superpixel size increases runtime.
    """)
    return


@app.cell
def _(mo):
    full_controls = mo.md(
        """
        **Full superpixel controls**

        Field point in frequency units: {field_freq}

        Superpixel size: {superpixel_size}

        Phase steps: {phase_steps}
        """
    ).batch(
        field_freq=mo.ui.slider(0.05, 0.25, step=0.01, value=0.15, show_value=True),
        superpixel_size=mo.ui.slider(100, 180, step=20, value=120, show_value=True),
        phase_steps=mo.ui.slider(1, 5, step=1, value=3, show_value=True),
    )
    full_run = mo.ui.run_button(label="Run compact full superpixel calibration")

    mo.vstack([full_controls, full_run], gap=1.0)
    return full_controls, full_run


@app.cell
def _(full_controls):
    full_field_freq = float(full_controls.value["field_freq"])
    full_phase_steps = int(full_controls.value["phase_steps"])
    full_superpixel_size = int(full_controls.value["superpixel_size"])
    return full_field_freq, full_phase_steps, full_superpixel_size


@app.cell
def _(
    collect_figures,
    full_field_freq,
    full_phase_steps,
    full_run,
    full_superpixel_size,
    mo,
    mp_fs,
    render_figures,
    scheduled_points_ij,
    warnings,
):
    mo.stop(
        not full_run.value,
        mo.md("Click **Run compact full superpixel calibration** to compute this section."),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        superpixel_calibration = mp_fs.wavefront_calibrate_superpixel(
            calibration_points=scheduled_points_ij,
            field_point=(full_field_freq, 0),
            field_point_units="freq",
            superpixel_size=full_superpixel_size,
            phase_steps=full_phase_steps,
            plot=-1,
        )

    _process_indices = sorted({0, scheduled_points_ij.shape[1] - 1})
    _, superpixel_process_figures = collect_figures(
        lambda: [
            mp_fs.wavefront_calibration_superpixel_process(
                index=process_index,
                r2_threshold=0.2,
                smooth=True,
                plot=True,
            )
            for process_index in _process_indices
        ]
    )

    mo.vstack(
        [
            mo.md(
                f"""
                Stored calibration key: `"wavefront_superpixel"`.
                Processed point indices: `{_process_indices}`.
                Calibration fields: `{sorted(superpixel_calibration.keys())}`.
                """
            ),
            render_figures(superpixel_process_figures),
        ],
        gap=1.0,
    )
    return superpixel_calibration


@app.cell
def _(mo):
    mo.md(r"""
    ## Multipoint Zernike Calibration

    The scheduled camera points are converted into Zernike coordinates, measured over a
    small perturbation sweep, smoothed, and then remeasured for a diagnostic plot.
    """)
    return


@app.cell
def _(mo):
    zernike_controls = mo.md(
        """
        **Zernike calibration controls**

        Highest ANSI index: {zernike_count}

        Perturbation span in pi: {perturbation_span}

        Perturbation steps: {perturbation_steps}

        Smoothing: {smoothing}
        """
    ).batch(
        zernike_count=mo.ui.slider(4, 10, step=1, value=6, show_value=True),
        perturbation_span=mo.ui.slider(0.5, 2.0, step=0.5, value=1.0, show_value=True),
        perturbation_steps=mo.ui.slider(3, 7, step=2, value=5, show_value=True),
        smoothing=mo.ui.slider(0.0, 1.0, step=0.1, value=0.5, show_value=True),
    )
    zernike_run = mo.ui.run_button(label="Run multipoint Zernike calibration")

    mo.vstack([zernike_controls, zernike_run], gap=1.0)
    return zernike_controls, zernike_run


@app.cell
def _(zernike_controls):
    perturbation_span_pi = float(zernike_controls.value["perturbation_span"])
    perturbation_steps = int(zernike_controls.value["perturbation_steps"])
    smoothing = float(zernike_controls.value["smoothing"])
    zernike_count = int(zernike_controls.value["zernike_count"])
    return perturbation_span_pi, perturbation_steps, smoothing, zernike_count


@app.cell
def _(
    collect_figures,
    mo,
    mp_fs,
    np,
    perturbation_span_pi,
    perturbation_steps,
    render_figures,
    scheduled_points_ij,
    smoothing,
    toolbox,
    warnings,
    zernike_count,
    zernike_run,
):
    mo.stop(
        not zernike_run.value,
        mo.md("Click **Run multipoint Zernike calibration** to compute this section."),
    )

    scheduled_points_zernike = toolbox.convert_vector(
        scheduled_points_ij,
        from_units="ij",
        to_units="zernike",
        hardware=mp_fs,
    )
    perturbations = np.linspace(
        -perturbation_span_pi * np.pi,
        perturbation_span_pi * np.pi,
        perturbation_steps,
        endpoint=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, initial_zernike_figures = collect_figures(
            lambda: mp_fs.wavefront_calibrate_zernike(
                scheduled_points_zernike,
                perturbation=0,
                plot=2,
            )
        )
        zernike_calibration = mp_fs.wavefront_calibrate_zernike(
            scheduled_points_zernike,
            zernike_indices=zernike_count,
            perturbation=perturbations,
            global_correction=True,
            plot=False,
        )
        smoothed_points_zernike, smooth_figures = collect_figures(
            lambda: mp_fs.wavefront_calibrate_zernike_smooth(
                smoothing=smoothing,
                smoothing_xy=smoothing,
                plot=True,
            )
        )
        _, final_zernike_figures = collect_figures(
            lambda: mp_fs.wavefront_calibrate_zernike(
                smoothed_points_zernike,
                zernike_indices=zernike_count,
                perturbation=0,
                plot=2,
            )
        )

    mo.vstack(
        [
            mo.md(
                f"""
                Stored calibration key: `"wavefront_zernike"`.
                Measured `{scheduled_points_ij.shape[1]}` points and smoothed to
                `{smoothed_points_zernike.shape[1]}` points.
                Calibration fields: `{sorted(zernike_calibration.keys())}`.
                """
            ),
            render_figures(initial_zernike_figures),
            render_figures(smooth_figures),
            render_figures(final_zernike_figures),
        ],
        gap=1.0,
    )
    return smoothed_points_zernike, zernike_calibration


@app.cell
def _(mo):
    mo.md(r"""
    ## What to Take Away

    - Multipoint wavefront calibration places several camera-space interference points at once.
    - The points must be spaced farther apart than the expected interference windows.
    - Superpixel calibration returns point-specific phase/amplitude corrections.
    - Zernike calibration can be measured at sparse points, smoothed, and remeasured.
    """)
    return


if __name__ == "__main__":
    app.run()
