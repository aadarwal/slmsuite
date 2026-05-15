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
    # Experimental Holography with Simulated Hardware

    This marimo notebook adapts the `slmsuite` Experimental Holography tutorial for a
    remote, simulation-first workflow. The original notebook demonstrates physical SLM
    and FLIR camera usage; here the same concepts run against `SimulatedSLM`,
    `SimulatedCamera`, and `FourierSLM`.

    The flow is:

    - initialize a camera-SLM system in a Fourier geometry
    - display flat and blazed phase patterns
    - calibrate the affine transform between camera pixels and SLM \(k\)-space
    - target a single camera pixel with a calibrated blaze
    - build square and scattered spot holograms
    - use simulated camera feedback to improve spot uniformity
    - form a small image with `FeedbackHologram`

    ![Experimental setup diagram](https://raw.githubusercontent.com/QPG-MIT/slmsuite-examples/main/examples/setup_diagram.png)
    """)
    return


@app.cell
def _():
    import contextlib
    import copy
    import io
    import os
    import sys
    import warnings
    from pathlib import Path

    import cv2
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.image import imread

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    mpl.rc("image", cmap="inferno")
    os.environ.setdefault("MPLBACKEND", "Agg")
    return contextlib, cv2, imread, io, np, plt, repo_root, warnings


@app.cell
def _(warnings):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cupy is not installed; using numpy.*")
        from slmsuite.hardware.cameras.simulated import SimulatedCamera
        from slmsuite.hardware.cameraslms import FourierSLM
        from slmsuite.hardware.slms.simulated import SimulatedSLM
        from slmsuite.holography import analysis, toolbox
        from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
    return (
        FeedbackHologram,
        FourierSLM,
        SimulatedCamera,
        SimulatedSLM,
        SpotHologram,
        analysis,
        toolbox,
    )


@app.cell
def _(contextlib, cv2, imread, io, mo, np, plt, repo_root, warnings):
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

    def capture_stdout(callback):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = callback()
        return result, stream.getvalue().strip()

    def logo_target(shape, width=220, x_shift=0):
        logo = imread(repo_root / "docs" / "source" / "static" / "slmsuite-small.png")
        gray = logo[..., :3].mean(axis=2) if logo.ndim == 3 else logo
        gray = 1 - gray
        height = max(1, int(width * gray.shape[0] / gray.shape[1]))
        small = cv2.resize(gray, (int(width), height), interpolation=cv2.INTER_AREA)
        target = np.zeros(shape)
        y0 = (target.shape[0] - small.shape[0]) // 2
        x0 = (target.shape[1] - small.shape[1]) // 2 + int(x_shift)
        y1 = min(target.shape[0], y0 + small.shape[0])
        x1 = min(target.shape[1], x0 + small.shape[1])
        sy0 = max(0, -y0)
        sx0 = max(0, -x0)
        y0 = max(0, y0)
        x0 = max(0, x0)
        target[y0:y1, x0:x1] = small[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
        return target

    def spot_power_summary(powers):
        powers_norm = powers / np.mean(powers)
        return float(np.std(powers_norm)), float(np.min(powers_norm)), float(np.max(powers_norm))

    return (
        capture_stdout,
        collect_figures,
        logo_target,
        render_figures,
        spot_power_summary,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Simulated Camera-SLM System

    `FourierSLM` is the composite object that owns a camera, an SLM, and calibrations
    connecting the SLM's normalized \(k\)-space to camera pixels. The controls below set
    up a compact simulated system that is fast enough to use interactively.
    """)
    return


@app.cell
def _(mo):
    exp_setup_controls = mo.md(
        """
        **Simulation setup**

        SLM width: {slm_width}

        Camera width: {camera_width}

        Camera-SLM rotation: {rotation_deg} degrees

        Camera gain: {gain}
        """
    ).batch(
        slm_width=mo.ui.slider(384, 768, step=128, value=512, show_value=True),
        camera_width=mo.ui.slider(288, 576, step=96, value=384, show_value=True),
        rotation_deg=mo.ui.slider(-10, 10, step=1, value=5, show_value=True),
        gain=mo.ui.slider(25, 200, step=25, value=50, show_value=True),
    )
    exp_setup_run = mo.ui.run_button(label="Reinitialize simulated hardware")

    mo.vstack([exp_setup_controls, exp_setup_run], gap=1.0)
    return exp_setup_controls, exp_setup_run


@app.cell
def _(exp_setup_controls):
    exp_camera_width = int(exp_setup_controls.value["camera_width"])
    exp_gain = float(exp_setup_controls.value["gain"])
    exp_rotation_deg = float(exp_setup_controls.value["rotation_deg"])
    exp_slm_width = int(exp_setup_controls.value["slm_width"])
    return exp_camera_width, exp_gain, exp_rotation_deg, exp_slm_width


@app.cell
def _(
    FourierSLM,
    SimulatedCamera,
    SimulatedSLM,
    exp_camera_width,
    exp_gain,
    exp_rotation_deg,
    exp_setup_run,
    exp_slm_width,
    np,
    warnings,
):
    exp_setup_run.value

    exp_slm_resolution = (exp_slm_width, int(round(exp_slm_width * 0.625)))
    exp_camera_resolution = (exp_camera_width, int(round(exp_camera_width * 0.75)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"'camera' _get_image_hw\(\) failed .*")
        exp_slm = SimulatedSLM(exp_slm_resolution, pitch_um=(8, 8))
        exp_slm.set_source_analytic(sim=True)
        exp_slm.set_source_analytic()
        exp_cam = SimulatedCamera(
            exp_slm,
            exp_camera_resolution,
            pitch_um=(4, 4),
            gain=exp_gain,
        )

    exp_fs = FourierSLM(exp_cam, exp_slm)
    exp_M, exp_b = exp_fs.fourier_calibration_build(
        f_eff=80000.0,
        theta=exp_rotation_deg * np.pi / 180,
    )
    exp_cam.set_affine(exp_M, exp_b)
    exp_fs.fourier_calibrate_analytic(exp_M, exp_b)
    exp_cam.set_exposure(29e-6)
    return exp_M, exp_b, exp_cam, exp_fs, exp_slm


@app.cell
def _(exp_M, exp_b, exp_cam, exp_slm, mo, np):
    mo.md(
        f"""
        Active simulated hardware:

        - SLM shape: `{exp_slm.shape}` pixels
        - Camera shape: `{exp_cam.shape}` pixels
        - Analytic Fourier calibration matrix:

        ```
        {np.array2string(exp_M, precision=3)}
        ```

        - Analytic camera offset: `{np.array2string(np.ravel(exp_b), precision=3)}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Simple Holography

    A flat phase sends light to the zero-th diffraction order. A blazed grating
    is a linear phase ramp, and its Fourier transform is a shifted spot in the
    camera plane.
    """)
    return


@app.cell
def _(mo):
    simple_controls = mo.md(
        r"""
        **Simple holography controls**

        Blaze \(k_x\) in mrad: {kx_mrad}

        Blaze \(k_y\) in mrad: {ky_mrad}

        Degree blaze: {deg_blaze}

        Target camera x pixel: {target_x}

        Target camera y pixel: {target_y}
        """
    ).batch(
        kx_mrad=mo.ui.slider(-5.0, 5.0, step=0.5, value=2.0, show_value=True),
        ky_mrad=mo.ui.slider(-5.0, 5.0, step=0.5, value=2.0, show_value=True),
        deg_blaze=mo.ui.slider(0.05, 0.4, step=0.05, value=0.2, show_value=True),
        target_x=mo.ui.slider(25, 350, step=25, value=200, show_value=True),
        target_y=mo.ui.slider(25, 250, step=25, value=150, show_value=True),
    )
    simple_run = mo.ui.run_button(label="Recompute simple holography")

    mo.vstack([simple_controls, simple_run], gap=1.0)
    return simple_controls, simple_run


@app.cell
def _(simple_controls):
    simple_deg_blaze = float(simple_controls.value["deg_blaze"])
    simple_kx = float(simple_controls.value["kx_mrad"]) * 1e-3
    simple_ky = float(simple_controls.value["ky_mrad"]) * 1e-3
    simple_target_x = int(simple_controls.value["target_x"])
    simple_target_y = int(simple_controls.value["target_y"])
    return (
        simple_deg_blaze,
        simple_kx,
        simple_ky,
        simple_target_x,
        simple_target_y,
    )


@app.cell
def _(
    capture_stdout,
    collect_figures,
    exp_cam,
    exp_fs,
    exp_slm,
    np,
    simple_deg_blaze,
    simple_kx,
    simple_ky,
    simple_run,
    simple_target_x,
    simple_target_y,
    toolbox,
):
    simple_run.value

    _, flat_figures = collect_figures(
        lambda: exp_fs.plot(
            np.zeros(exp_slm.shape),
            title="Flat phase: zero-th order",
            cam_limits=0.5,
        )
    )

    simple_vector = (simple_kx, simple_ky)
    simple_blaze_phase = toolbox.phase.blaze(grid=exp_slm, vector=simple_vector)
    _, blaze_figures = collect_figures(
        lambda: exp_fs.plot(
            simple_blaze_phase,
            title=f"Blaze at {simple_vector}",
            cam_limits=0.5,
        )
    )

    _, simple_conversion_text = capture_stdout(
        lambda: toolbox.print_blaze_conversions(
            vector=simple_vector,
            from_units="norm",
            hardware=exp_slm,
        )
    )

    degree_vector = toolbox.convert_vector(
        (simple_deg_blaze, simple_deg_blaze),
        from_units="deg",
        to_units="norm",
    )
    degree_blaze_phase = toolbox.phase.blaze(grid=exp_slm, vector=degree_vector)
    _, degree_figures = collect_figures(
        lambda: exp_fs.plot(
            degree_blaze_phase,
            title=f"Blaze at {simple_deg_blaze:.2f} degrees",
            cam_limits=0.5,
        )
    )

    target_pixel = (
        min(simple_target_x, exp_cam.shape[1] - 1),
        min(simple_target_y, exp_cam.shape[0] - 1),
    )
    target_vector = exp_fs.ijcam_to_kxyslm(target_pixel)
    target_phase = toolbox.phase.blaze(grid=exp_slm, vector=target_vector)
    _, target_figures = collect_figures(
        lambda: exp_fs.plot(
            target_phase,
            title=f"Calibrated blaze to camera pixel {target_pixel}",
            cam_limits=0.5,
        )
    )

    _, calibrated_conversion_text = capture_stdout(
        lambda: toolbox.print_blaze_conversions(
            vector=target_vector,
            from_units="norm",
            hardware=exp_fs,
        )
    )
    return (
        blaze_figures,
        calibrated_conversion_text,
        degree_figures,
        flat_figures,
        simple_conversion_text,
        target_figures,
        target_pixel,
        target_vector,
    )


@app.cell
def _(
    blaze_figures,
    calibrated_conversion_text,
    degree_figures,
    flat_figures,
    mo,
    render_figures,
    simple_conversion_text,
    target_figures,
    target_pixel,
    target_vector,
):
    mo.vstack(
        [
            render_figures(flat_figures),
            render_figures(blaze_figures),
            mo.md(f"Raw SLM-only blaze conversions:\n\n```text\n{simple_conversion_text}\n```"),
            render_figures(degree_figures),
            mo.md(
                f"After Fourier calibration, camera pixel `{target_pixel}` maps to normalized "
                f"`k` vector `{target_vector.ravel()}`."
            ),
            render_figures(target_figures),
            mo.md(f"Calibrated conversions:\n\n```text\n{calibrated_conversion_text}\n```"),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fourier Calibration

    The setup starts with an analytic calibration so the rest of the notebook is usable
    immediately. The button below runs the same camera-feedback calibration routine used
    in the physical tutorial: it displays a spot grid, fits the camera positions, and
    replaces the affine calibration.
    """)
    return


@app.cell
def _(mo):
    fourier_controls = mo.md(
        """
        **Fourier calibration controls**

        Calibration grid side length: {array_shape}

        Calibration pitch in `"knm"` pixels: {array_pitch}
        """
    ).batch(
        array_shape=mo.ui.slider(3, 9, step=2, value=5, show_value=True),
        array_pitch=mo.ui.slider(6, 18, step=2, value=10, show_value=True),
    )
    fourier_run = mo.ui.run_button(label="Run Fourier calibration")

    mo.vstack([fourier_controls, fourier_run], gap=1.0)
    return fourier_controls, fourier_run


@app.cell
def _(fourier_controls):
    fourier_array_pitch = int(fourier_controls.value["array_pitch"])
    fourier_array_shape = int(fourier_controls.value["array_shape"])
    return fourier_array_pitch, fourier_array_shape


@app.cell
def _(
    collect_figures,
    exp_fs,
    fourier_array_pitch,
    fourier_array_shape,
    fourier_run,
    mo,
    np,
    render_figures,
    warnings,
):
    if fourier_run.value:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, fourier_figures = collect_figures(
                lambda: exp_fs.fourier_calibrate(
                    array_shape=fourier_array_shape,
                    array_pitch=fourier_array_pitch,
                    plot=True,
                )
            )
        fourier_status = "Camera-feedback Fourier calibration was run."
    else:
        fourier_figures = []
        fourier_status = "Using the analytic Fourier calibration from simulated setup."

    fourier_calibration = exp_fs.calibrations["fourier"]
    mo.vstack(
        [
            mo.md(
                f"""
                {fourier_status}

                Current affine matrix:

                ```
                {np.array2string(fourier_calibration["M"], precision=3)}
                ```

                Current offset: `{np.array2string(np.ravel(fourier_calibration["b"]), precision=3)}`
                """
            ),
            render_figures(fourier_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Square Spot Holography

    `SpotHologram` uses the Fourier calibration to accept target positions in camera
    `"ij"` pixels. This section first optimizes computationally, then optionally uses
    simulated camera feedback to make the spot powers more uniform.
    """)
    return


@app.cell
def _(mo):
    square_controls = mo.md(
        """
        **Square spot-array controls**

        Camera-pixel pitch: {pitch}

        Margin from camera edge: {margin}

        Hologram computational side: {holo_side}

        Computational WGS iterations: {comp_iters}

        Experimental feedback iterations: {exp_iters}
        """
    ).batch(
        pitch=mo.ui.slider(50, 120, step=10, value=80, show_value=True),
        margin=mo.ui.slider(25, 80, step=5, value=40, show_value=True),
        holo_side=mo.ui.slider(512, 1024, step=512, value=512, show_value=True),
        comp_iters=mo.ui.slider(3, 20, step=1, value=8, show_value=True),
        exp_iters=mo.ui.slider(0, 10, step=1, value=4, show_value=True),
    )
    square_run = mo.ui.run_button(label="Run square spot holography")

    mo.vstack([square_controls, square_run], gap=1.0)
    return square_controls, square_run


@app.cell
def _(square_controls):
    square_comp_iters = int(square_controls.value["comp_iters"])
    square_exp_iters = int(square_controls.value["exp_iters"])
    square_holo_side = int(square_controls.value["holo_side"])
    square_margin = int(square_controls.value["margin"])
    square_pitch = int(square_controls.value["pitch"])
    return (
        square_comp_iters,
        square_exp_iters,
        square_holo_side,
        square_margin,
        square_pitch,
    )


@app.cell
def _(
    SpotHologram,
    analysis,
    collect_figures,
    exp_fs,
    mo,
    np,
    plt,
    render_figures,
    spot_power_summary,
    square_comp_iters,
    square_exp_iters,
    square_holo_side,
    square_margin,
    square_pitch,
    square_run,
    warnings,
):
    mo.stop(not square_run.value, mo.md("Click **Run square spot holography** to compute this section."))

    xlist = np.arange(square_margin, min(exp_fs.cam.shape) - square_margin, square_pitch)
    square_xgrid, square_ygrid = np.meshgrid(xlist, xlist)
    square_points = np.vstack((square_xgrid.ravel(), square_ygrid.ravel()))

    def plot_square_points():
        plt.figure(figsize=(6, 5))
        plt.scatter(square_points[0], square_points[1])
        plt.xlim([0, exp_fs.cam.shape[1]])
        plt.ylim([exp_fs.cam.shape[0], 0])
        plt.xlabel("Camera x [pix]")
        plt.ylabel("Camera y [pix]")
        plt.title("Requested square spot array")

    _, square_point_figures = collect_figures(plot_square_points)

    square_hologram = SpotHologram(
        shape=(square_holo_side, square_holo_side),
        spot_vectors=square_points,
        basis="ij",
        cameraslm=exp_fs,
    )
    square_limits = [
        [square_holo_side // 2 - 140, square_holo_side // 2 + 140],
        [square_holo_side // 2 - 140, square_holo_side // 2 + 140],
    ]
    _, square_before_figures = collect_figures(
        lambda: (
            square_hologram.plot_farfield(
                square_hologram.target,
                limits=square_limits,
                title="Target",
            ),
            square_hologram.plot_farfield(
                limits=square_limits,
                title="Before optimization",
            ),
        )
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        square_hologram.optimize(
            "WGS-Kim",
            maxiter=square_comp_iters,
            feedback="computational_spot",
            stat_groups=["computational_spot"],
            verbose=False,
        )

    _, square_after_figures = collect_figures(
        lambda: (
            square_hologram.plot_farfield(
                limits=square_limits,
                title="After computational WGS",
            ),
            square_hologram.plot_nearfield(title="Nearfield phase"),
        )
    )

    exp_fs.slm.set_phase(square_hologram.get_phase(), settle=False)
    square_img_comp = exp_fs.cam.get_image()
    square_subimages_comp = analysis.take(square_img_comp, vectors=square_points, size=9)
    square_powers_comp = analysis.image_normalization(square_subimages_comp)
    square_std_comp, square_min_comp, square_max_comp = spot_power_summary(square_powers_comp)

    if square_exp_iters > 0:
        square_hologram.spot_integration_width_ij = 9
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            square_hologram.optimize(
                "WGS-Kim",
                maxiter=square_exp_iters,
                feedback="experimental_spot",
                stat_groups=["computational_spot", "experimental_spot"],
                fixed_phase=False,
                verbose=False,
            )
        square_feedback_note = "Experimental spot feedback was applied."
    else:
        square_feedback_note = "Experimental spot feedback was skipped."

    square_img_final = exp_fs.cam.get_image()
    square_subimages_final = analysis.take(square_img_final, vectors=square_points, size=9)
    square_powers_final = analysis.image_normalization(square_subimages_final)
    square_std_final, square_min_final, square_max_final = spot_power_summary(square_powers_final)

    def plot_square_camera_and_histograms():
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        exp_fs.cam.plot(square_img_final, ax=axs[0])
        axs[0].scatter(square_points[0], square_points[1], 90, facecolors="none", edgecolors="r")
        axs[0].set_title("Simulated camera result")
        axs[1].hist(square_powers_comp / np.mean(square_powers_comp))
        axs[1].set_title(f"Before feedback std={100 * square_std_comp:.1f}%")
        axs[2].hist(square_powers_final / np.mean(square_powers_final))
        axs[2].set_title(f"Final std={100 * square_std_final:.1f}%")
        fig.tight_layout()

    _, square_camera_figures = collect_figures(plot_square_camera_and_histograms)
    _, square_stats_figures = collect_figures(lambda: square_hologram.plot_stats())

    mo.vstack(
        [
            mo.md(
                f"""
                Requested `{square_points.shape[1]}` spots.

                {square_feedback_note}

                Spot-power summary:

                - computational-only normalized std: `{square_std_comp:.3f}` with range `{square_min_comp:.3f}` to `{square_max_comp:.3f}`
                - final normalized std: `{square_std_final:.3f}` with range `{square_min_final:.3f}` to `{square_max_final:.3f}`
                """
            ),
            render_figures(square_point_figures),
            render_figures(square_before_figures),
            render_figures(square_after_figures),
            render_figures(square_camera_figures),
            render_figures(square_stats_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scattered Spot Holography

    A square grid is highly structured. Lloyd's algorithm gives a more scattered pattern,
    which often optimizes faster because the target has less crystalline symmetry.
    """)
    return


@app.cell
def _(mo):
    scatter_controls = mo.md(
        """
        **Scattered spot controls**

        Number of requested points: {n_points}

        Lloyd iterations: {lloyd_iters}

        WGS iterations: {wgs_iters}
        """
    ).batch(
        n_points=mo.ui.slider(20, 100, step=10, value=50, show_value=True),
        lloyd_iters=mo.ui.slider(5, 50, step=5, value=20, show_value=True),
        wgs_iters=mo.ui.slider(3, 20, step=1, value=8, show_value=True),
    )
    scatter_run = mo.ui.run_button(label="Run scattered spot holography")

    mo.vstack([scatter_controls, scatter_run], gap=1.0)
    return scatter_controls, scatter_run


@app.cell
def _(scatter_controls):
    scatter_lloyd_iters = int(scatter_controls.value["lloyd_iters"])
    scatter_n_points = int(scatter_controls.value["n_points"])
    scatter_wgs_iters = int(scatter_controls.value["wgs_iters"])
    return scatter_lloyd_iters, scatter_n_points, scatter_wgs_iters


@app.cell
def _(
    SpotHologram,
    collect_figures,
    exp_fs,
    mo,
    np,
    plt,
    render_figures,
    scatter_lloyd_iters,
    scatter_n_points,
    scatter_run,
    scatter_wgs_iters,
    toolbox,
    warnings,
):
    mo.stop(not scatter_run.value, mo.md("Click **Run scattered spot holography** to compute this section."))

    scatter_points = toolbox.lloyds_points(
        grid=exp_fs.cam.shape,
        n_points=scatter_n_points,
        iterations=scatter_lloyd_iters,
    )
    zeroth_order_ij = exp_fs.kxyslm_to_ijcam((0, 0))
    difference_vectors = scatter_points - zeroth_order_ij
    distance_to_zeroth = np.sqrt(np.square(difference_vectors[0]) + np.square(difference_vectors[1]))
    scatter_points = np.delete(scatter_points, int(np.argmin(distance_to_zeroth)), axis=1)

    def plot_scattered_points():
        plt.figure(figsize=(6, 5))
        plt.scatter(scatter_points[0], scatter_points[1], alpha=0.7)
        plt.scatter(zeroth_order_ij[0], zeroth_order_ij[1], c="r", marker="x", label="0th order")
        plt.xlim([0, exp_fs.cam.shape[1]])
        plt.ylim([exp_fs.cam.shape[0], 0])
        plt.legend()
        plt.title("Scattered camera targets")

    _, scatter_point_figures = collect_figures(plot_scattered_points)

    scatter_hologram = SpotHologram(
        (512, 512),
        scatter_points,
        basis="ij",
        cameraslm=exp_fs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scatter_hologram.optimize(
            "WGS-Kim",
            maxiter=scatter_wgs_iters,
            feedback="computational_spot",
            stat_groups=["computational_spot"],
            verbose=False,
        )
    exp_fs.slm.set_phase(scatter_hologram.get_phase(), settle=False)
    scatter_img = exp_fs.cam.get_image()

    def plot_scattered_camera():
        plt.figure(figsize=(7, 7))
        ax = exp_fs.cam.plot(scatter_img)
        ax.scatter(scatter_points[0], scatter_points[1], 180, facecolors="none", edgecolors="r")

    _, scatter_camera_figures = collect_figures(plot_scattered_camera)
    _, scatter_stats_figures = collect_figures(lambda: scatter_hologram.plot_stats())

    mo.vstack(
        [
            mo.md(f"Optimized `{scatter_points.shape[1]}` scattered targets."),
            render_figures(scatter_point_figures),
            render_figures(scatter_camera_figures),
            render_figures(scatter_stats_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pictorial Holography

    `FeedbackHologram` generalizes the target from isolated spots to an image in the
    camera plane. The target image is embedded in camera `"ij"` coordinates, transformed
    into computational `"knm"` coordinates, and optimized.
    """)
    return


@app.cell
def _(mo):
    image_controls = mo.md(
        """
        **Image holography controls**

        Logo width in camera pixels: {logo_width}

        Computational WGS iterations: {comp_iters}

        Experimental feedback iterations: {exp_iters}
        """
    ).batch(
        logo_width=mo.ui.slider(120, 300, step=20, value=220, show_value=True),
        comp_iters=mo.ui.slider(2, 15, step=1, value=5, show_value=True),
        exp_iters=mo.ui.slider(0, 8, step=1, value=3, show_value=True),
    )
    image_run = mo.ui.run_button(label="Run image holography")

    mo.vstack([image_controls, image_run], gap=1.0)
    return image_controls, image_run


@app.cell
def _(image_controls):
    exp_image_comp_iters = int(image_controls.value["comp_iters"])
    exp_image_exp_iters = int(image_controls.value["exp_iters"])
    exp_image_logo_width = int(image_controls.value["logo_width"])
    return exp_image_comp_iters, exp_image_exp_iters, exp_image_logo_width


@app.cell
def _(
    FeedbackHologram,
    collect_figures,
    exp_fs,
    exp_image_comp_iters,
    exp_image_exp_iters,
    exp_image_logo_width,
    image_run,
    logo_target,
    mo,
    render_figures,
    warnings,
):
    mo.stop(not image_run.value, mo.md("Click **Run image holography** to compute this section."))

    image_target_ij = logo_target(exp_fs.cam.shape, width=exp_image_logo_width)
    image_hologram = FeedbackHologram(
        shape=(512, 512),
        target_ij=image_target_ij,
        cameraslm=exp_fs,
    )
    image_limits, image_target_figures = collect_figures(
        lambda: image_hologram.plot_farfield(image_hologram.target)
    )
    image_hologram.reset_phase(quadratic_phase=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        image_hologram.optimize(
            method="WGS-Leonardo",
            maxiter=exp_image_comp_iters,
            feedback="computational",
            stat_groups=["computational"],
            verbose=False,
        )

    exp_fs.slm.set_phase(image_hologram.get_phase(), settle=False)
    image_cam_comp = exp_fs.cam.get_image()
    _, image_comp_figures = collect_figures(
        lambda: (
            exp_fs.cam.plot(image_target_ij, title="Target image"),
            exp_fs.cam.plot(image_cam_comp, title="Simulated camera after computational WGS"),
            image_hologram.plot_nearfield(title="Image hologram nearfield"),
            image_hologram.plot_farfield(limits=image_limits, title="Computational farfield"),
        )
    )

    if exp_image_exp_iters > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            image_hologram.optimize(
                method="WGS-Leonardo",
                maxiter=exp_image_exp_iters,
                feedback="experimental",
                stat_groups=["computational", "experimental"],
                blur_ij=1,
                verbose=False,
            )
        image_feedback_note = "Experimental image feedback was applied."
    else:
        image_feedback_note = "Experimental image feedback was skipped."

    exp_fs.slm.set_phase(image_hologram.get_phase(), settle=False)
    image_cam_final = exp_fs.cam.get_image()
    _, image_final_figures = collect_figures(
        lambda: exp_fs.cam.plot(image_cam_final, title="Final simulated camera image")
    )

    mo.vstack(
        [
            mo.md(image_feedback_note),
            render_figures(image_target_figures),
            render_figures(image_comp_figures, columns=2),
            render_figures(image_final_figures),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Takeaways

    - `FourierSLM` is the bridge between camera pixels and SLM spatial-frequency coordinates.
    - A blaze moves one diffraction order; spot and image holography require iterative phase retrieval.
    - `SpotHologram` adds spot-specific targets, integration windows, and statistics.
    - Camera feedback can reweight the optimization using measured power instead of only the computational model.
    - `FeedbackHologram` extends the same feedback loop to dense image targets.
    """)
    return


if __name__ == "__main__":
    app.run()
