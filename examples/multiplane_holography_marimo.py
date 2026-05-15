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
    # Multiplane Holography with `slmsuite`

    This marimo notebook adapts the multiplane holography documentation example to a
    compact simulated system. It combines image objectives at different propagation
    planes with compressed Zernike spot objectives, then optimizes the shared SLM phase
    through `MultiplaneHologram`.
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
        from slmsuite.holography.algorithms import (
            CompressedSpotHologram,
            FeedbackHologram,
            MultiplaneHologram,
        )
        from slmsuite.holography.analysis.files import _load_image as load_image_file
        from slmsuite.holography.toolbox import phase as phase_tools

    return (
        CompressedSpotHologram,
        FeedbackHologram,
        FourierSLM,
        MultiplaneHologram,
        SimulatedCamera,
        SimulatedSLM,
        load_image_file,
        phase_tools,
        toolbox,
    )


@app.cell
def _(mo, np, plt, repo_root):
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

    def image_path(name):
        return repo_root / "docs" / "source" / "static" / name

    def load_logo_target(loader, camera_shape, name, target_shape, shift=(0, 0)):
        return loader(str(image_path(name)), camera_shape, target_shape=target_shape, shift=shift)

    def plot_targets_and_kernels(camera, slm, targets, kernels, labels):
        fig, axs = plt.subplots(2, len(targets), figsize=(4.5 * len(targets), 7))
        if len(targets) == 1:
            axs = np.array(axs).reshape(2, 1)
        for index, (target, kernel, label) in enumerate(zip(targets, kernels, labels)):
            camera.plot(target, title=f"{label} target", ax=axs[0, index], cbar=False)
            slm.plot(kernel if kernel is not None else np.zeros(slm.shape), title=f"{label} kernel", ax=axs[1, index], cbar=False)
        fig.tight_layout()
        return axs

    def plot_spot_map(camera_shape, spots_ij, colors, title, color_label):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(np.zeros(camera_shape), cmap="gray", vmin=0, vmax=1)
        scatter = ax.scatter(
            spots_ij[0],
            spots_ij[1],
            c=colors,
            cmap="coolwarm",
            s=55,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_title(title)
        ax.set_xlim(0, camera_shape[1] - 1)
        ax.set_ylim(camera_shape[0] - 1, 0)
        ax.set_xlabel("Camera i [pix]")
        ax.set_ylabel("Camera j [pix]")
        fig.colorbar(scatter, ax=ax, label=color_label)
        return ax

    return (
        collect_figures,
        load_logo_target,
        plot_spot_map,
        plot_targets_and_kernels,
        render_figures,
    )


@app.cell
def _(mo):
    setup_controls = mo.md(
        """
        **Multiplane simulation setup**

        SLM width: {slm_width}

        Camera width: {camera_width}

        Camera gain: {gain}
        """
    ).batch(
        slm_width=mo.ui.slider(128, 320, step=32, value=160, show_value=True),
        camera_width=mo.ui.slider(160, 320, step=20, value=180, show_value=True),
        gain=mo.ui.slider(40, 160, step=20, value=80, show_value=True),
    )
    setup_run = mo.ui.run_button(label="Reinitialize multiplane simulation")

    mo.vstack([setup_controls, setup_run], gap=1.0)
    return setup_controls, setup_run


@app.cell
def _(setup_controls):
    ml_camera_width = int(setup_controls.value["camera_width"])
    ml_gain = float(setup_controls.value["gain"])
    ml_slm_width = int(setup_controls.value["slm_width"])
    return ml_camera_width, ml_gain, ml_slm_width


@app.cell
def _(
    FourierSLM,
    SimulatedCamera,
    SimulatedSLM,
    ml_camera_width,
    ml_gain,
    ml_slm_width,
    np,
    setup_run,
    warnings,
):
    setup_run.value

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"'camera' _get_image_hw\(\) failed .*")
        ml_slm = SimulatedSLM((ml_slm_width, ml_slm_width), pitch_um=(8, 8))
        ml_slm.set_source_analytic(sim=True)
        ml_slm.set_source_analytic()
        ml_cam = SimulatedCamera(
            ml_slm,
            (ml_camera_width, ml_camera_width),
            pitch_um=(4, 4),
            gain=ml_gain,
        )

    ml_fs = FourierSLM(ml_cam, ml_slm)
    ml_M, ml_b = ml_fs.fourier_calibration_build(
        f_eff=60000.0,
        theta=5 * np.pi / 180,
    )
    ml_cam.set_affine(ml_M, ml_b)
    ml_fs.fourier_calibrate_analytic(ml_M, ml_b)
    ml_cam.set_exposure(0.02)
    return ml_M, ml_b, ml_cam, ml_fs, ml_slm


@app.cell
def _(ml_M, ml_b, ml_cam, ml_slm, mo, np):
    mo.md(
        f"""
        Active simulated system:

        - SLM shape: `{ml_slm.shape}` pixels
        - Camera shape: `{ml_cam.shape}` pixels
        - Fourier matrix determinant: `{np.linalg.det(ml_M):.3g}`
        - Fourier offset: `{np.array2string(np.ravel(ml_b), precision=3)}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Target Preview

    The example uses the small `slmsuite` image assets as camera-plane amplitude targets.
    Different propagation kernels are then assigned to each target plane.
    """)
    return


@app.cell
def _(mo):
    target_controls = mo.md(
        """
        **Target controls**

        Target height: {target_height}

        Target width: {target_width}

        Vertical shift: {vertical_shift}
        """
    ).batch(
        target_height=mo.ui.slider(40, 120, step=10, value=70, show_value=True),
        target_width=mo.ui.slider(80, 180, step=10, value=140, show_value=True),
        vertical_shift=mo.ui.slider(-30, 30, step=5, value=0, show_value=True),
    )
    target_run = mo.ui.run_button(label="Reload preview targets")

    mo.vstack([target_controls, target_run], gap=1.0)
    return target_controls, target_run


@app.cell
def _(target_controls):
    ml_target_height = int(target_controls.value["target_height"])
    ml_target_width = int(target_controls.value["target_width"])
    ml_vertical_shift = int(target_controls.value["vertical_shift"])
    return ml_target_height, ml_target_width, ml_vertical_shift


@app.cell
def _(
    load_image_file,
    collect_figures,
    load_logo_target,
    ml_cam,
    ml_slm,
    ml_target_height,
    ml_target_width,
    ml_vertical_shift,
    np,
    phase_tools,
    plot_targets_and_kernels,
    target_run,
):
    target_run.value

    ml_focus_f = 1e7
    ml_target_shape = (ml_target_height, ml_target_width)
    ml_target_shift = (ml_vertical_shift, 0)
    ml_focus_labels = ["near plane", "middle plane", "far plane"]
    ml_focus_files = [
        "slmsuite-small-smith.png",
        "slmsuite-small-spin.png",
        "slmsuite-small-text.png",
    ]
    ml_focus_depths = [-ml_focus_f, np.inf, ml_focus_f]
    ml_focus_targets = [
        load_logo_target(load_image_file, ml_cam.shape, filename, ml_target_shape, shift=ml_target_shift)
        for filename in ml_focus_files
    ]
    ml_focus_kernels = [
        None if np.isinf(focal_length) else phase_tools.lens(ml_slm, f=focal_length)
        for focal_length in ml_focus_depths
    ]

    _, target_preview_figures = collect_figures(
        lambda: plot_targets_and_kernels(
            ml_cam,
            ml_slm,
            ml_focus_targets,
            ml_focus_kernels,
            ml_focus_labels,
        )
    )
    return (
        ml_focus_depths,
        ml_focus_files,
        ml_focus_kernels,
        ml_focus_labels,
        ml_focus_targets,
        ml_target_shape,
        ml_target_shift,
        target_preview_figures,
    )


@app.cell
def _(render_figures, target_preview_figures):
    render_figures(target_preview_figures)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Planes of Focus

    Each `FeedbackHologram` objective gets a different propagation kernel. The parent
    `MultiplaneHologram` optimizes one shared nearfield phase against all planes.
    """)
    return


@app.cell
def _(mo):
    focus_controls = mo.md(
        """
        **Focus-plane optimization controls**

        GS iterations: {gs_iterations}

        WGS iterations: {wgs_iterations}

        Null-region radius fraction: {null_radius}
        """
    ).batch(
        gs_iterations=mo.ui.slider(1, 8, step=1, value=2, show_value=True),
        wgs_iterations=mo.ui.slider(1, 12, step=1, value=4, show_value=True),
        null_radius=mo.ui.slider(0.30, 0.60, step=0.05, value=0.45, show_value=True),
    )
    focus_run = mo.ui.run_button(label="Run planes-of-focus optimization")

    mo.vstack([focus_controls, focus_run], gap=1.0)
    return focus_controls, focus_run


@app.cell
def _(focus_controls):
    focus_gs_iterations = int(focus_controls.value["gs_iterations"])
    focus_null_radius = float(focus_controls.value["null_radius"])
    focus_wgs_iterations = int(focus_controls.value["wgs_iterations"])
    return focus_gs_iterations, focus_null_radius, focus_wgs_iterations


@app.cell
def _(
    FeedbackHologram,
    MultiplaneHologram,
    collect_figures,
    focus_gs_iterations,
    focus_null_radius,
    focus_run,
    focus_wgs_iterations,
    ml_focus_kernels,
    ml_focus_labels,
    ml_focus_targets,
    ml_fs,
    ml_slm,
    mo,
    np,
    render_figures,
):
    mo.stop(
        not focus_run.value,
        mo.md("Click **Run planes-of-focus optimization** to compute this section."),
    )

    focus_holograms = []
    focus_weights = []
    for _target, _kernel in zip(ml_focus_targets, ml_focus_kernels):
        focus_holograms.append(
            FeedbackHologram(
                ml_slm.shape,
                _target,
                cameraslm=ml_fs,
                propagation_kernel=_kernel,
                null_region_radius_frac=focus_null_radius,
            )
        )
        focus_weights.append(np.sqrt(np.nansum(np.square(_target.astype(float)))))

    focus_multiplane_hologram = MultiplaneHologram(focus_holograms, weights=focus_weights)
    focus_multiplane_hologram.reset_phase(random_phase=0, quadratic_phase=2)
    focus_multiplane_hologram.optimize(
        method="GS",
        maxiter=focus_gs_iterations,
        mraf_factor=1,
        verbose=False,
    )
    focus_multiplane_hologram.optimize(
        method="WGS-Leonardo",
        maxiter=focus_wgs_iterations,
        mraf_factor=0.5,
        feedback="computational",
        verbose=False,
    )

    _, focus_result_figures = collect_figures(
        lambda: (
            ml_slm.plot(focus_multiplane_hologram.get_phase(), title="Shared multiplane phase"),
            focus_multiplane_hologram.plot_farfield(),
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
                Optimized `{len(focus_holograms)}` image planes: `{ml_focus_labels}`.
                Relative objective weights: `{np.array(focus_weights).round(2).tolist()}`.
                """
            ),
            render_figures(focus_result_figures),
        ],
        gap=1.0,
    )
    return focus_multiplane_hologram


@app.cell
def _(mo):
    mo.md(r"""
    ## Planes of Basis

    A dense camera-plane image can be optimized alongside a compressed spot hologram.
    The spot hologram below uses a higher-order Zernike basis for spot-specific aberrations.
    """)
    return


@app.cell
def _(mo):
    basis_controls = mo.md(
        """
        **Basis-plane controls**

        Spot count: {spot_count}

        Aberration span: {aberration_span}

        WGS iterations: {wgs_iterations}
        """
    ).batch(
        spot_count=mo.ui.slider(4, 14, step=2, value=8, show_value=True),
        aberration_span=mo.ui.slider(0.0, 3.0, step=0.5, value=1.5, show_value=True),
        wgs_iterations=mo.ui.slider(1, 12, step=1, value=4, show_value=True),
    )
    basis_run = mo.ui.run_button(label="Run image-plus-basis optimization")

    mo.vstack([basis_controls, basis_run], gap=1.0)
    return basis_controls, basis_run


@app.cell
def _(basis_controls):
    basis_aberration_span = float(basis_controls.value["aberration_span"])
    basis_spot_count = int(basis_controls.value["spot_count"])
    basis_wgs_iterations = int(basis_controls.value["wgs_iterations"])
    return basis_aberration_span, basis_spot_count, basis_wgs_iterations


@app.cell
def _(
    CompressedSpotHologram,
    FeedbackHologram,
    MultiplaneHologram,
    load_image_file,
    basis_aberration_span,
    basis_run,
    basis_spot_count,
    basis_wgs_iterations,
    collect_figures,
    load_logo_target,
    ml_cam,
    ml_fs,
    ml_slm,
    ml_target_shape,
    ml_target_shift,
    mo,
    np,
    plot_spot_map,
    render_figures,
    toolbox,
):
    mo.stop(
        not basis_run.value,
        mo.md("Click **Run image-plus-basis optimization** to compute this section."),
    )

    basis_image_target = load_logo_target(
        load_image_file,
        ml_cam.shape,
        "slmsuite-small.png",
        ml_target_shape,
        shift=ml_target_shift,
    )
    basis_image_hologram = FeedbackHologram(
        ml_slm.shape,
        basis_image_target,
        cameraslm=ml_fs,
        null_region_radius_frac=0.45,
    )

    _theta = np.linspace(0, 2 * np.pi, basis_spot_count, endpoint=False)
    _radius = 0.26 * min(ml_cam.shape)
    basis_spots_ij = np.vstack(
        (
            ml_cam.shape[1] / 2 + _radius * np.cos(_theta),
            ml_cam.shape[0] / 2 + _radius * np.sin(_theta),
        )
    )
    _basis_lateral_zernike = toolbox.convert_vector(
        basis_spots_ij,
        from_units="ij",
        to_units="zernike",
        hardware=ml_fs,
    )
    _basis_sample = np.linspace(-1, 1, basis_spot_count)
    basis_indices = [2, 1, 4, 3, 5, 6]
    basis_vectors = np.vstack(
        (
            _basis_lateral_zernike,
            basis_aberration_span * _basis_sample,
            basis_aberration_span * np.sin(np.pi * _basis_sample),
            basis_aberration_span * np.cos(np.pi * _basis_sample),
            0.5 * basis_aberration_span * _basis_sample**2,
        )
    )
    basis_spot_hologram = CompressedSpotHologram(
        spot_vectors=basis_vectors,
        basis=basis_indices,
        cameraslm=ml_fs,
    )

    basis_multiplane_hologram = MultiplaneHologram(
        [basis_image_hologram, basis_spot_hologram],
        weights=[1.0, 0.5],
    )
    basis_multiplane_hologram.reset_phase(random_phase=0, quadratic_phase=2)
    basis_multiplane_hologram.optimize(
        method="WGS-Leonardo",
        maxiter=basis_wgs_iterations,
        mraf_factor=0.8,
        feedback="computational",
        verbose=False,
    )

    _, basis_result_figures = collect_figures(
        lambda: (
            plot_spot_map(
                ml_cam.shape,
                basis_spots_ij,
                basis_vectors[2],
                "Compressed basis spots colored by focus",
                "ANSI 4 coefficient",
            ),
            ml_slm.plot(basis_multiplane_hologram.get_phase(), title="Image + basis phase"),
            basis_multiplane_hologram.plot_farfield(),
        )
    )

    mo.vstack(
        [
            mo.md(
                f"Optimized one image objective with `{basis_spot_count}` compressed spots in basis `{basis_indices}`."
            ),
            render_figures(basis_result_figures),
        ],
        gap=1.0,
    )
    return basis_multiplane_hologram


@app.cell
def _(mo):
    mo.md(r"""
    ## Planes of Focus and Basis

    The final compact workflow combines three image planes with a compressed 3D spot field.
    The spot field uses a vortex-waveplate basis entry (`-1`) to mimic bottle-beam style
    spot shaping.
    """)
    return


@app.cell
def _(mo):
    combined_controls = mo.md(
        """
        **Combined optimization controls**

        Starfield spots: {spot_count}

        Depth span: {depth_span}

        GS iterations: {gs_iterations}

        WGS iterations: {wgs_iterations}
        """
    ).batch(
        spot_count=mo.ui.slider(6, 30, step=2, value=12, show_value=True),
        depth_span=mo.ui.slider(0.25, 2.0, step=0.25, value=1.0, show_value=True),
        gs_iterations=mo.ui.slider(1, 6, step=1, value=1, show_value=True),
        wgs_iterations=mo.ui.slider(1, 12, step=1, value=4, show_value=True),
    )
    combined_run = mo.ui.run_button(label="Run combined multiplane optimization")

    mo.vstack([combined_controls, combined_run], gap=1.0)
    return combined_controls, combined_run


@app.cell
def _(combined_controls):
    combined_depth_span = float(combined_controls.value["depth_span"])
    combined_gs_iterations = int(combined_controls.value["gs_iterations"])
    combined_spot_count = int(combined_controls.value["spot_count"])
    combined_wgs_iterations = int(combined_controls.value["wgs_iterations"])
    return combined_depth_span, combined_gs_iterations, combined_spot_count, combined_wgs_iterations


@app.cell
def _(
    CompressedSpotHologram,
    FeedbackHologram,
    MultiplaneHologram,
    collect_figures,
    combined_depth_span,
    combined_gs_iterations,
    combined_run,
    combined_spot_count,
    combined_wgs_iterations,
    ml_cam,
    ml_focus_kernels,
    ml_focus_targets,
    ml_fs,
    ml_slm,
    mo,
    np,
    plot_spot_map,
    render_figures,
    toolbox,
):
    mo.stop(
        not combined_run.value,
        mo.md("Click **Run combined multiplane optimization** to compute this section."),
    )

    combined_holograms = [
        FeedbackHologram(
            ml_slm.shape,
            _target,
            cameraslm=ml_fs,
            propagation_kernel=_kernel,
            null_region_radius_frac=0.45,
        )
        for _target, _kernel in zip(ml_focus_targets, ml_focus_kernels)
    ]
    combined_weights = [
        np.sqrt(np.nansum(np.square(_target.astype(float))))
        for _target in ml_focus_targets
    ]

    _rng = np.random.default_rng(14)
    combined_spots_ij = np.vstack(
        (
            _rng.uniform(0.22 * ml_cam.shape[1], 0.78 * ml_cam.shape[1], combined_spot_count),
            _rng.uniform(0.22 * ml_cam.shape[0], 0.78 * ml_cam.shape[0], combined_spot_count),
        )
    )
    combined_spots_kxy = toolbox.convert_vector(
        combined_spots_ij,
        from_units="ij",
        to_units="kxy",
        hardware=ml_fs,
    )
    combined_depth_coefficients = (
        combined_depth_span * 1e-7 * _rng.uniform(-1, 1, combined_spot_count)
    )
    combined_spots_zernike = toolbox.convert_vector(
        np.vstack((combined_spots_kxy, combined_depth_coefficients)),
        from_units="kxy",
        to_units="zernike",
        hardware=ml_fs,
    )
    combined_spots_zernike = np.vstack(
        (
            combined_spots_zernike,
            2 * np.ones(combined_spot_count),
        )
    )
    combined_basis = [2, 1, 4, -1]
    combined_spot_hologram = CompressedSpotHologram(
        spot_vectors=combined_spots_zernike,
        basis=combined_basis,
        cameraslm=ml_fs,
    )
    combined_holograms.append(combined_spot_hologram)
    combined_weights.append(0.4 * np.mean(combined_weights))

    combined_multiplane_hologram = MultiplaneHologram(
        combined_holograms,
        weights=combined_weights,
    )
    combined_multiplane_hologram.reset_phase(random_phase=0, quadratic_phase=2)
    combined_multiplane_hologram.optimize(
        method="GS",
        maxiter=combined_gs_iterations,
        mraf_factor=1,
        verbose=False,
    )
    combined_multiplane_hologram.optimize(
        method="WGS-Leonardo",
        maxiter=combined_wgs_iterations,
        mraf_factor=0.8,
        feedback="computational",
        verbose=False,
    )

    _, combined_result_figures = collect_figures(
        lambda: (
            plot_spot_map(
                ml_cam.shape,
                combined_spots_ij,
                combined_depth_coefficients,
                "Compressed starfield spots colored by depth",
                "depth coefficient",
            ),
            ml_slm.plot(combined_multiplane_hologram.get_phase(), title="Combined multiplane phase"),
            combined_multiplane_hologram.plot_farfield(),
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
                Optimized `{len(combined_holograms)}` objectives:
                three image planes plus `{combined_spot_count}` compressed 3D spots.
                """
            ),
            render_figures(combined_result_figures),
        ],
        gap=1.0,
    )
    return combined_multiplane_hologram


@app.cell
def _(mo):
    mo.md(r"""
    ## What to Take Away

    - `MultiplaneHologram` shares one SLM phase across several child hologram objectives.
    - `FeedbackHologram` handles dense camera-plane image targets and optional propagation kernels.
    - `CompressedSpotHologram` can be mixed into the same parent objective for sparse spot fields.
    - Weights rebalance power between image planes and spot objectives after each child target normalizes itself.
    """)
    return


if __name__ == "__main__":
    app.run()
