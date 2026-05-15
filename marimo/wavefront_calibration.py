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
    # Wavefront Calibration with Simulated Hardware

    The computational and experimental holography tutorials assume an ideal Fourier
    relationship and a known nearfield source. Real systems usually violate both:
    optics add aberration, and the source amplitude is not uniform.

    This marimo notebook adapts the `slmsuite` wavefront calibration example to a
    compact simulated system. It demonstrates:

    - an aberrated SLM source
    - uncorrected flat and blazed diffraction spots
    - Fourier calibration as a prerequisite for wavefront calibration
    - single-superpixel calibration testing
    - compact full superpixel calibration
    - processing the measured superpixel data into per-pixel phase/amplitude corrections
    - why Fourier calibration is normally rerun after wavefront correction
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
    return np, plt, warnings


@app.cell
def _(warnings):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cupy is not installed; using numpy.*")
        from slmsuite.hardware.cameras.simulated import SimulatedCamera
        from slmsuite.hardware.cameraslms import FourierSLM
        from slmsuite.hardware.slms.simulated import SimulatedSLM
        from slmsuite.holography.toolbox.phase import blaze, zernike_sum
    return FourierSLM, SimulatedCamera, SimulatedSLM, blaze, zernike_sum


@app.cell
def _(mo, np, plt):
    def collect_figures(plotter):
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        for number in plt.get_fignums():
            plt.close(number)
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

    def plot_camera_slm(fs, title="", cam_limits=0.35):
        _, axs = plt.subplots(1, 2, figsize=(12, 4))
        fs.slm.plot(fs.slm.phase, title="Displayed phase", ax=axs[0], cbar=True)
        fs.cam.plot(title="Simulated camera", limits=cam_limits, ax=axs[1], cbar=True)
        plt.suptitle(title)
        plt.tight_layout()

    def render_frame_strip(frames, max_frames=4):
        if frames is None or len(frames) == 0:
            return mo.md("_No movie frames were generated._")
        indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)

        def plot_frames():
            fig, axs = plt.subplots(1, len(indices), figsize=(4 * len(indices), 3))
            if len(indices) == 1:
                axs = [axs]
            for ax, index in zip(axs, indices):
                ax.imshow(frames[index])
                ax.set_title(f"phase frame {index + 1}")
                ax.axis("off")
            fig.tight_layout()

        _, figures = collect_figures(plot_frames)
        return render_figures(figures)

    return collect_figures, plot_camera_slm, render_figures, render_frame_strip


@app.cell
def _(mo):
    mo.md(r"""
    ## Initialization

    The simulated SLM below is intentionally given a Gaussian source with a Zernike-like
    phase aberration. The camera is placed in a Fourier geometry and starts with an
    analytic Fourier calibration so coordinate conversion works before we optionally
    run the camera-feedback calibration routine.
    """)
    return


@app.cell
def _(mo):
    wf_setup_controls = mo.md(
        """
        **Wavefront simulation setup**

        SLM width: {slm_width}

        Camera width: {camera_width}

        Aberration scale: {aberration_scale}

        Camera gain: {gain}
        """
    ).batch(
        slm_width=mo.ui.slider(240, 480, step=80, value=320, show_value=True),
        camera_width=mo.ui.slider(220, 440, step=68, value=288, show_value=True),
        aberration_scale=mo.ui.slider(0.25, 2.0, step=0.25, value=1.0, show_value=True),
        gain=mo.ui.slider(50, 300, step=50, value=200, show_value=True),
    )
    wf_setup_run = mo.ui.run_button(label="Reinitialize wavefront simulation")

    mo.vstack([wf_setup_controls, wf_setup_run], gap=1.0)
    return wf_setup_controls, wf_setup_run


@app.cell
def _(wf_setup_controls):
    wf_aberration_scale = float(wf_setup_controls.value["aberration_scale"])
    wf_camera_width = int(wf_setup_controls.value["camera_width"])
    wf_gain = float(wf_setup_controls.value["gain"])
    wf_slm_width = int(wf_setup_controls.value["slm_width"])
    return wf_aberration_scale, wf_camera_width, wf_gain, wf_slm_width


@app.cell
def _(
    FourierSLM,
    SimulatedCamera,
    SimulatedSLM,
    collect_figures,
    np,
    warnings,
    wf_aberration_scale,
    wf_camera_width,
    wf_gain,
    wf_setup_run,
    wf_slm_width,
    zernike_sum,
):
    wf_setup_run.value

    wf_slm_resolution = (wf_slm_width, int(round(wf_slm_width * 0.75)))
    wf_camera_resolution = (wf_camera_width, int(round(wf_camera_width * 0.76)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"'camera' _get_image_hw\(\) failed .*")
        wf_slm = SimulatedSLM(wf_slm_resolution, pitch_um=(8, 8))
        wf_phase_aberration = wf_aberration_scale * zernike_sum(
            wf_slm,
            indices=(3, 4, 5, 7, 8),
            weights=(1, -2, 3, 1, 1),
            aperture=None,
            use_mask=False,
        )
        wf_slm.set_source_analytic(phase_offset=wf_phase_aberration, sim=True)
        wf_cam = SimulatedCamera(
            wf_slm,
            wf_camera_resolution,
            pitch_um=(4, 4),
            gain=wf_gain,
        )

    wf_fs = FourierSLM(wf_cam, wf_slm)
    wf_M, wf_b = wf_fs.fourier_calibration_build(
        f_eff=80000.0,
        theta=5 * np.pi / 180,
    )
    wf_cam.set_affine(wf_M, wf_b)
    wf_fs.fourier_calibrate_analytic(wf_M, wf_b)
    wf_cam.set_exposure(0.04)

    _, wf_source_figures = collect_figures(lambda: wf_slm.plot_source(sim=True))
    return wf_M, wf_b, wf_cam, wf_fs, wf_slm, wf_source_figures


@app.cell
def _(mo, np, render_figures, wf_M, wf_b, wf_cam, wf_slm, wf_source_figures):
    mo.vstack(
        [
            mo.md(
                f"""
                Active simulated system:

                - SLM shape: `{wf_slm.shape}` pixels
                - Camera shape: `{wf_cam.shape}` pixels
                - Starting Fourier matrix:

                ```
                {np.array2string(wf_M, precision=3)}
                ```

                - Starting Fourier offset: `{np.array2string(np.ravel(wf_b), precision=3)}`
                """
            ),
            render_figures(wf_source_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Without Wavefront Calibration

    Before calibration, the simulated source has phase curvature and a nonuniform
    amplitude. A flat display and a simple blaze therefore produce broadened,
    distorted camera spots.
    """)
    return


@app.cell
def _(mo):
    wf_probe_controls = mo.md(
        r"""
        **Uncorrected probe controls**

        Blaze \(k_x\) in mrad: {kx_mrad}

        Blaze \(k_y\) in mrad: {ky_mrad}

        Camera zoom fraction: {cam_limits}
        """
    ).batch(
        kx_mrad=mo.ui.slider(-5.0, 5.0, step=0.5, value=2.0, show_value=True),
        ky_mrad=mo.ui.slider(-5.0, 5.0, step=0.5, value=2.0, show_value=True),
        cam_limits=mo.ui.slider(0.15, 0.6, step=0.05, value=0.35, show_value=True),
    )
    wf_probe_run = mo.ui.run_button(label="Recompute uncorrected probes")

    mo.vstack([wf_probe_controls, wf_probe_run], gap=1.0)
    return wf_probe_controls, wf_probe_run


@app.cell
def _(wf_probe_controls):
    wf_cam_limits = float(wf_probe_controls.value["cam_limits"])
    wf_probe_kx = float(wf_probe_controls.value["kx_mrad"]) * 1e-3
    wf_probe_ky = float(wf_probe_controls.value["ky_mrad"]) * 1e-3
    return wf_cam_limits, wf_probe_kx, wf_probe_ky


@app.cell
def _(
    blaze,
    collect_figures,
    mo,
    plot_camera_slm,
    render_figures,
    wf_cam_limits,
    wf_fs,
    wf_probe_kx,
    wf_probe_ky,
    wf_probe_run,
    wf_slm,
):
    wf_probe_run.value

    wf_slm.set_phase(None, settle=False)
    _, wf_flat_figures = collect_figures(
        lambda: plot_camera_slm(
            wf_fs,
            title="No correction, no blaze",
            cam_limits=wf_cam_limits,
        )
    )

    wf_uncorrected_blaze = blaze(grid=wf_slm, vector=(wf_probe_kx, wf_probe_ky))
    wf_slm.set_phase(wf_uncorrected_blaze, settle=False)
    _, wf_blaze_figures = collect_figures(
        lambda: plot_camera_slm(
            wf_fs,
            title="No correction, with blaze",
            cam_limits=wf_cam_limits,
        )
    )

    mo.vstack(
        [
            render_figures(wf_flat_figures),
            render_figures(wf_blaze_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fourier Calibration

    Wavefront calibration needs a Fourier calibration because it must place
    calibration and field diffraction orders at known camera points. The simulated
    setup starts with an analytic calibration; this button runs a measured calibration
    grid like the original tutorial.
    """)
    return


@app.cell
def _(mo):
    wf_fourier_controls = mo.md(
        """
        **Fourier calibration controls**

        Grid side length: {array_shape}

        Grid pitch in `"knm"` pixels: {array_pitch}
        """
    ).batch(
        array_shape=mo.ui.slider(3, 9, step=2, value=5, show_value=True),
        array_pitch=mo.ui.slider(6, 18, step=2, value=8, show_value=True),
    )
    wf_fourier_run = mo.ui.run_button(label="Run wavefront prerequisite Fourier calibration")

    mo.vstack([wf_fourier_controls, wf_fourier_run], gap=1.0)
    return wf_fourier_controls, wf_fourier_run


@app.cell
def _(wf_fourier_controls):
    wf_fourier_array_pitch = int(wf_fourier_controls.value["array_pitch"])
    wf_fourier_array_shape = int(wf_fourier_controls.value["array_shape"])
    return wf_fourier_array_pitch, wf_fourier_array_shape


@app.cell
def _(
    collect_figures,
    mo,
    np,
    render_figures,
    warnings,
    wf_fourier_array_pitch,
    wf_fourier_array_shape,
    wf_fourier_run,
    wf_fs,
):
    if wf_fourier_run.value:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, wf_fourier_figures = collect_figures(
                lambda: wf_fs.fourier_calibrate(
                    array_shape=wf_fourier_array_shape,
                    array_pitch=wf_fourier_array_pitch,
                    plot=True,
                )
            )
        wf_fourier_note = "Camera-feedback Fourier calibration was run."
    else:
        wf_fourier_figures = []
        wf_fourier_note = "Using the analytic Fourier calibration from setup."

    wf_fourier_calibration = wf_fs.calibrations["fourier"]
    mo.vstack(
        [
            mo.md(
                f"""
                {wf_fourier_note}

                Current Fourier offset: `{np.array2string(np.ravel(wf_fourier_calibration["b"]), precision=3)}`
                """
            ),
            render_figures(wf_fourier_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Testing Wavefront Calibration

    `wavefront_calibrate()` can test a single scheduled superpixel before running the
    full serial sweep. In the original notebook `plot=3` generates a phase-sweep movie.
    Here the frames are shown as a strip so the app remains lightweight.
    """)
    return


@app.cell
def _(mo):
    wf_test_controls = mo.md(
        """
        **Single-superpixel test controls**

        Calibration x pixel: {cal_x}

        Calibration y pixel: {cal_y}

        Field point in frequency units: {field_freq}

        Superpixel size: {superpixel_size}

        Test index: {test_index}

        Phase steps: {phase_steps}
        """
    ).batch(
        cal_x=mo.ui.slider(50, 260, step=10, value=160, show_value=True),
        cal_y=mo.ui.slider(40, 200, step=10, value=100, show_value=True),
        field_freq=mo.ui.slider(0.05, 0.25, step=0.01, value=0.12, show_value=True),
        superpixel_size=mo.ui.slider(40, 120, step=10, value=80, show_value=True),
        test_index=mo.ui.slider(0, 8, step=1, value=2, show_value=True),
        phase_steps=mo.ui.slider(3, 8, step=1, value=4, show_value=True),
    )
    wf_test_run = mo.ui.run_button(label="Run single-superpixel wavefront test")

    mo.vstack([wf_test_controls, wf_test_run], gap=1.0)
    return wf_test_controls, wf_test_run


@app.cell
def _(wf_test_controls):
    wf_test_cal_x = int(wf_test_controls.value["cal_x"])
    wf_test_cal_y = int(wf_test_controls.value["cal_y"])
    wf_test_field_freq = float(wf_test_controls.value["field_freq"])
    wf_test_index = int(wf_test_controls.value["test_index"])
    wf_test_phase_steps = int(wf_test_controls.value["phase_steps"])
    wf_test_superpixel_size = int(wf_test_controls.value["superpixel_size"])
    return (
        wf_test_cal_x,
        wf_test_cal_y,
        wf_test_field_freq,
        wf_test_index,
        wf_test_phase_steps,
        wf_test_superpixel_size,
    )


@app.cell
def _(
    mo,
    render_frame_strip,
    warnings,
    wf_cam,
    wf_fs,
    wf_test_cal_x,
    wf_test_cal_y,
    wf_test_field_freq,
    wf_test_index,
    wf_test_phase_steps,
    wf_test_run,
    wf_test_superpixel_size,
):
    mo.stop(
        not wf_test_run.value,
        mo.md("Click **Run single-superpixel wavefront test** to compute this section."),
    )

    wf_test_point = (
        min(wf_test_cal_x, wf_cam.shape[1] - 1),
        min(wf_test_cal_y, wf_cam.shape[0] - 1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf_test_frames = wf_fs.wavefront_calibrate(
            calibration_points=wf_test_point,
            field_point=(wf_test_field_freq, 0),
            field_point_units="freq",
            superpixel_size=wf_test_superpixel_size,
            corrected_amplitude=True,
            test_index=wf_test_index,
            phase_steps=wf_test_phase_steps,
            plot=3,
        )

    mo.vstack(
        [
            mo.md(
                f"""
                Tested schedule index `{wf_test_index}` at camera point `{wf_test_point}`.

                The frames show the SLM reference/test superpixels and the corresponding camera
                interference region as phase is swept.
                """
            ),
            render_frame_strip(wf_test_frames),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Full Compact Wavefront Calibration

    A full physical calibration can take many minutes. The compact simulation below
    uses a small SLM and large superpixels so the full serial sweep finishes quickly
    while preserving the data flow: measure raw superpixel calibration, process it into
    phase/amplitude maps, and apply the correction.
    """)
    return


@app.cell
def _(mo):
    wf_full_controls = mo.md(
        r"""
        **Compact full calibration controls**

        Calibration x pixel: {cal_x}

        Calibration y pixel: {cal_y}

        Field point in frequency units: {field_freq}

        Superpixel size: {superpixel_size}

        Phase steps: {phase_steps}

        Processing \(R^2\) threshold: {r2_threshold}
        """
    ).batch(
        cal_x=mo.ui.slider(50, 260, step=10, value=160, show_value=True),
        cal_y=mo.ui.slider(40, 200, step=10, value=100, show_value=True),
        field_freq=mo.ui.slider(0.05, 0.25, step=0.01, value=0.12, show_value=True),
        superpixel_size=mo.ui.slider(80, 160, step=20, value=120, show_value=True),
        phase_steps=mo.ui.slider(1, 6, step=1, value=5, show_value=True),
        r2_threshold=mo.ui.slider(0.0, 0.9, step=0.1, value=0.2, show_value=True),
    )
    wf_full_run = mo.ui.run_button(label="Run compact full wavefront calibration")

    mo.vstack([wf_full_controls, wf_full_run], gap=1.0)
    return wf_full_controls, wf_full_run


@app.cell
def _(wf_full_controls):
    wf_full_cal_x = int(wf_full_controls.value["cal_x"])
    wf_full_cal_y = int(wf_full_controls.value["cal_y"])
    wf_full_field_freq = float(wf_full_controls.value["field_freq"])
    wf_full_phase_steps = int(wf_full_controls.value["phase_steps"])
    wf_full_r2_threshold = float(wf_full_controls.value["r2_threshold"])
    wf_full_superpixel_size = int(wf_full_controls.value["superpixel_size"])
    return (
        wf_full_cal_x,
        wf_full_cal_y,
        wf_full_field_freq,
        wf_full_phase_steps,
        wf_full_r2_threshold,
        wf_full_superpixel_size,
    )


@app.cell
def _(
    blaze,
    collect_figures,
    mo,
    np,
    plot_camera_slm,
    render_figures,
    warnings,
    wf_cam,
    wf_cam_limits,
    wf_fs,
    wf_full_cal_x,
    wf_full_cal_y,
    wf_full_field_freq,
    wf_full_phase_steps,
    wf_full_r2_threshold,
    wf_full_run,
    wf_full_superpixel_size,
    wf_probe_kx,
    wf_probe_ky,
    wf_slm,
):
    mo.stop(
        not wf_full_run.value,
        mo.md("Click **Run compact full wavefront calibration** to compute this section."),
    )

    wf_full_point = (
        min(wf_full_cal_x, wf_cam.shape[1] - 1),
        min(wf_full_cal_y, wf_cam.shape[0] - 1),
    )
    wf_b_before = np.array(wf_fs.calibrations["fourier"]["b"], copy=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf_raw_calibration = wf_fs.wavefront_calibrate(
            calibration_points=wf_full_point,
            field_point=(wf_full_field_freq, 0),
            field_point_units="freq",
            superpixel_size=wf_full_superpixel_size,
            phase_steps=wf_full_phase_steps,
            plot=-1,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, wf_process_unsmoothed_figures = collect_figures(
            lambda: wf_fs.wavefront_calibration_superpixel_process(
                r2_threshold=wf_full_r2_threshold,
                smooth=False,
                plot=True,
            )
        )
        _, wf_process_smoothed_figures = collect_figures(
            lambda: wf_fs.wavefront_calibration_superpixel_process(
                r2_threshold=wf_full_r2_threshold,
                smooth=True,
                plot=True,
                remove_background=True,
            )
        )

    _, wf_phase_figures = collect_figures(
        lambda: wf_slm.plot(wf_slm.source["phase"], title="Processed wavefront phase")
    )

    wf_slm.set_phase(None, settle=False)
    _, wf_corrected_flat_figures = collect_figures(
        lambda: plot_camera_slm(
            wf_fs,
            title="With correction, no blaze",
            cam_limits=wf_cam_limits,
        )
    )

    wf_slm.set_phase(blaze(grid=wf_slm, vector=(wf_probe_kx, wf_probe_ky)), settle=False)
    _, wf_corrected_blaze_figures = collect_figures(
        lambda: plot_camera_slm(
            wf_fs,
            title="With correction, with blaze",
            cam_limits=wf_cam_limits,
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, wf_recalibration_figures = collect_figures(
            lambda: wf_fs.fourier_calibrate(
                array_shape=5,
                array_pitch=8,
                plot=True,
            )
        )
    wf_b_after = np.array(wf_fs.calibrations["fourier"]["b"], copy=True)

    mo.vstack(
        [
            mo.md(
                f"""
                Raw calibration supershape: `{wf_raw_calibration["slm_supershape"]}`.

                Fourier offset before wavefront correction: `{np.array2string(np.ravel(wf_b_before), precision=3)}`

                Fourier offset after rerunning calibration: `{np.array2string(np.ravel(wf_b_after), precision=3)}`

                Offset shift: `{np.array2string(np.ravel(wf_b_after - wf_b_before), precision=3)}`
                """
            ),
            render_figures(wf_process_unsmoothed_figures),
            render_figures(wf_process_smoothed_figures),
            render_figures(wf_phase_figures),
            render_figures(wf_corrected_flat_figures),
            render_figures(wf_corrected_blaze_figures),
            render_figures(wf_recalibration_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Takeaways

    - Wavefront calibration measures source amplitude and phase across SLM superpixels.
    - The test mode is for tuning calibration point, field point, exposure, superpixel size, and phase steps.
    - Processing converts sparse superpixel measurements into full-resolution source maps.
    - The resulting phase map compensates optical aberration when `SLM.set_phase()` is called.
    - Fourier calibration should usually be repeated after wavefront correction, because the correction can shift spot positions.
    """)
    return


if __name__ == "__main__":
    app.run()
