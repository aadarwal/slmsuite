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
    # Zernike Holography with `slmsuite`

    This marimo notebook converts the Zernike holography documentation example into
    interactive sections. It covers Zernike indexing, derivative-aware phase plots,
    weighted Zernike sums, and compressed spot holograms whose spot positions are
    represented directly in a Zernike basis.
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
        from slmsuite.holography.algorithms import CompressedSpotHologram
        from slmsuite.holography.toolbox import phase as phase_tools

    return (
        CompressedSpotHologram,
        FourierSLM,
        SimulatedCamera,
        SimulatedSLM,
        phase_tools,
        toolbox,
    )


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

    def plot_phase_series(slm, entries):
        axes = []
        for title, phase_data in entries:
            plt.figure(figsize=(8, 4))
            axes.append(slm.plot(phase_data, title=title))
        return axes

    def plot_spot_map(camera_shape, spot_ij, colors, title, color_label):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(np.zeros(camera_shape), cmap="gray", vmin=0, vmax=1)
        scatter = ax.scatter(
            spot_ij[0],
            spot_ij[1],
            c=colors,
            cmap="coolwarm",
            s=60,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel("Camera i [pix]")
        ax.set_ylabel("Camera j [pix]")
        ax.set_xlim(0, camera_shape[1] - 1)
        ax.set_ylim(camera_shape[0] - 1, 0)
        fig.colorbar(scatter, ax=ax, label=color_label)
        return ax

    def plot_coefficient_matrix(matrix, row_labels, title):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("Spot index")
        ax.set_ylabel("Zernike basis")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        return ax

    return (
        collect_figures,
        plot_coefficient_matrix,
        plot_phase_series,
        plot_spot_map,
        render_figures,
    )


@app.cell
def _(mo):
    setup_controls = mo.md(
        """
        **Simulated hardware**

        SLM width: {slm_width}

        Camera width: {camera_width}

        Fourier rotation: {rotation}
        """
    ).batch(
        slm_width=mo.ui.slider(128, 384, step=64, value=192, show_value=True),
        camera_width=mo.ui.slider(160, 384, step=32, value=220, show_value=True),
        rotation=mo.ui.slider(-8.0, 8.0, step=1.0, value=0.0, show_value=True),
    )
    setup_run = mo.ui.run_button(label="Rebuild simulated hardware")

    mo.vstack([setup_controls, setup_run], gap=1.0)
    return setup_controls, setup_run


@app.cell
def _(setup_controls):
    camera_width = int(setup_controls.value["camera_width"])
    rotation_deg = float(setup_controls.value["rotation"])
    slm_width = int(setup_controls.value["slm_width"])
    return camera_width, rotation_deg, slm_width


@app.cell
def _(
    FourierSLM,
    SimulatedCamera,
    SimulatedSLM,
    camera_width,
    np,
    rotation_deg,
    setup_run,
    slm_width,
    warnings,
):
    setup_run.value

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"'camera' _get_image_hw\(\) failed .*")
        z_slm = SimulatedSLM((slm_width, slm_width), pitch_um=(8, 8))
        z_slm.set_source_analytic(sim=True)
        z_slm.set_source_analytic()
        z_cam = SimulatedCamera(
            z_slm,
            (camera_width, camera_width),
            pitch_um=(4, 4),
            gain=1,
        )

    z_fs = FourierSLM(z_cam, z_slm)
    z_M, z_b = z_fs.fourier_calibration_build(
        f_eff=80000.0,
        theta=rotation_deg * np.pi / 180,
    )
    z_cam.set_affine(z_M, z_b)
    z_fs.fourier_calibrate_analytic(z_M, z_b)
    z_cam.set_exposure(0.03)
    return z_b, z_cam, z_fs, z_M, z_slm


@app.cell
def _(mo, np, z_b, z_cam, z_M, z_slm):
    mo.md(
        f"""
        Active simulation:

        - SLM shape: `{z_slm.shape}` pixels
        - Camera shape: `{z_cam.shape}` pixels
        - Fourier matrix determinant: `{np.linalg.det(z_M):.3g}`
        - Camera offset: `{np.squeeze(z_b).round(2).tolist()}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Zernike Indexing and Derivatives

    `slmsuite` uses ANSI Zernike indexing internally. The conversion helper maps between
    ANSI singleton indices and radial/azimuthal pairs, and the same derivative controls
    can be passed into `zernike`, `zernike_sum`, and `zernike_pyramid_plot`.
    """)
    return


@app.cell
def _(mo):
    index_controls = mo.md(
        """
        **Index and pyramid controls**

        Maximum ANSI index in table: {max_index}

        Pyramid radial order: {pyramid_order}

        Derivative dx: {derivative_x}

        Derivative dy: {derivative_y}
        """
    ).batch(
        max_index=mo.ui.slider(5, 24, step=1, value=14, show_value=True),
        pyramid_order=mo.ui.slider(2, 6, step=1, value=4, show_value=True),
        derivative_x=mo.ui.slider(0, 2, step=1, value=0, show_value=True),
        derivative_y=mo.ui.slider(0, 2, step=1, value=0, show_value=True),
    )
    index_run = mo.ui.run_button(label="Render Zernike index plots")

    mo.vstack([index_controls, index_run], gap=1.0)
    return index_controls, index_run


@app.cell
def _(index_controls):
    derivative_x = int(index_controls.value["derivative_x"])
    derivative_y = int(index_controls.value["derivative_y"])
    max_index = int(index_controls.value["max_index"])
    pyramid_order = int(index_controls.value["pyramid_order"])
    return derivative_x, derivative_y, max_index, pyramid_order


@app.cell
def _(
    collect_figures,
    derivative_x,
    derivative_y,
    index_run,
    max_index,
    np,
    phase_tools,
    pyramid_order,
    z_slm,
):
    index_run.value

    ansi_indices = np.arange(max_index + 1)
    radial_indices = phase_tools.zernike_convert_index(
        ansi_indices,
        from_index="ansi",
        to_index="radial",
    )
    zernike_rows = [
        "| ANSI | radial n | azimuthal l | polynomial |",
        "| ---: | ---: | ---: | :--- |",
    ]
    for ansi_index, (radial_n, azimuthal_l) in zip(ansi_indices, radial_indices):
        formula = phase_tools.zernike_get_string(int(ansi_index)).replace("|", "\\|")
        zernike_rows.append(
            f"| {ansi_index} | {radial_n} | {azimuthal_l} | `{formula}` |"
        )
    zernike_index_table = "\n".join(zernike_rows)

    derivative = (derivative_x, derivative_y)
    _, zernike_pyramid_figures = collect_figures(
        lambda: phase_tools.zernike_pyramid_plot(
            z_slm,
            order=pyramid_order,
            derivative=derivative,
            use_mask=False,
            titles=["ansi", "name"],
        )
    )
    return zernike_index_table, zernike_pyramid_figures


@app.cell
def _(mo, render_figures, zernike_index_table, zernike_pyramid_figures):
    mo.vstack(
        [
            mo.md(zernike_index_table),
            render_figures(zernike_pyramid_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Weighted Zernike Sums

    Weighted sums combine aberration modes into a single phase correction. The sliders
    below separate the selected single mode from a fixed, documentation-style sum.
    """)
    return


@app.cell
def _(mo):
    sum_controls = mo.md(
        """
        **Zernike sum controls**

        Single ANSI index: {single_index}

        Single weight: {single_weight}

        Sum scale: {sum_scale}
        """
    ).batch(
        single_index=mo.ui.slider(0, 24, step=1, value=8, show_value=True),
        single_weight=mo.ui.slider(-4.0, 4.0, step=0.5, value=2.0, show_value=True),
        sum_scale=mo.ui.slider(0.25, 3.0, step=0.25, value=1.0, show_value=True),
    )
    sum_run = mo.ui.run_button(label="Render Zernike sums")

    mo.vstack([sum_controls, sum_run], gap=1.0)
    return sum_controls, sum_run


@app.cell
def _(sum_controls):
    single_index = int(sum_controls.value["single_index"])
    single_weight = float(sum_controls.value["single_weight"])
    sum_scale = float(sum_controls.value["sum_scale"])
    return single_index, single_weight, sum_scale


@app.cell
def _(
    collect_figures,
    phase_tools,
    plot_phase_series,
    single_index,
    single_weight,
    sum_run,
    sum_scale,
    z_slm,
):
    sum_run.value

    single_phase = phase_tools.zernike(
        z_slm,
        index=single_index,
        weight=single_weight,
        use_mask=False,
    )
    fixed_indices = (3, 4, 5, 7, 8, 12)
    fixed_weights = tuple(sum_scale * value for value in (1.0, -2.0, 1.5, 0.75, -0.5, 0.4))
    sum_phase = phase_tools.zernike_sum(
        z_slm,
        indices=fixed_indices,
        weights=fixed_weights,
        use_mask=False,
    )

    _, zernike_sum_figures = collect_figures(
        lambda: plot_phase_series(
            z_slm,
            [
                (f"Single ANSI Zernike {single_index}", single_phase),
                ("Weighted Zernike sum", sum_phase),
                ("Sum plus weak tilt", sum_phase + phase_tools.blaze(z_slm, vector=(0.001, -0.001))),
            ],
        )
    )
    return fixed_indices, fixed_weights, zernike_sum_figures


@app.cell
def _(fixed_indices, fixed_weights, mo, render_figures, zernike_sum_figures):
    mo.vstack(
        [
            mo.md(f"Fixed sum uses ANSI indices `{fixed_indices}` with weights `{fixed_weights}`."),
            render_figures(zernike_sum_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3D Compressed Spot Holography

    `CompressedSpotHologram` stores only the nonzero spot targets. Here the lateral camera
    positions are converted into the Zernike tilt basis, and the third coefficient is
    ANSI Zernike 4, the focus term.
    """)
    return


@app.cell
def _(mo):
    spot_controls = mo.md(
        """
        **Compressed spot controls**

        Spot count: {spot_count}

        Lateral radius: {spot_radius}

        Focus coefficient span: {focus_span}

        WGS iterations: {spot_iterations}
        """
    ).batch(
        spot_count=mo.ui.slider(3, 15, step=2, value=7, show_value=True),
        spot_radius=mo.ui.slider(12.0, 55.0, step=4.0, value=32.0, show_value=True),
        focus_span=mo.ui.slider(0.0, 4.0, step=0.5, value=2.0, show_value=True),
        spot_iterations=mo.ui.slider(1, 15, step=1, value=5, show_value=True),
    )
    spot_run = mo.ui.run_button(label="Run compressed spot hologram")

    mo.vstack([spot_controls, spot_run], gap=1.0)
    return spot_controls, spot_run


@app.cell
def _(spot_controls):
    focus_span = float(spot_controls.value["focus_span"])
    spot_count = int(spot_controls.value["spot_count"])
    spot_iterations = int(spot_controls.value["spot_iterations"])
    spot_radius = float(spot_controls.value["spot_radius"])
    return focus_span, spot_count, spot_iterations, spot_radius


@app.cell
def _(
    CompressedSpotHologram,
    collect_figures,
    focus_span,
    mo,
    np,
    plot_coefficient_matrix,
    plot_spot_map,
    render_figures,
    spot_count,
    spot_iterations,
    spot_radius,
    spot_run,
    toolbox,
    z_cam,
    z_fs,
):
    mo.stop(
        not spot_run.value,
        mo.md("Click **Run compressed spot hologram** to compute this section."),
    )

    theta = np.linspace(0, 2 * np.pi, spot_count, endpoint=False)
    radial = np.linspace(0.25, 1.0, spot_count) * spot_radius
    _spot_ij = np.vstack(
        (
            z_cam.shape[1] / 2 + radial * np.cos(theta),
            z_cam.shape[0] / 2 + radial * np.sin(theta),
        )
    )
    _lateral_zernike = toolbox.convert_vector(
        _spot_ij,
        from_units="ij",
        to_units="zernike",
        hardware=z_fs,
    )
    focus_coefficients = np.linspace(-focus_span, focus_span, spot_count)
    spot_zernike = np.vstack((_lateral_zernike, focus_coefficients))

    compressed_spot_hologram = CompressedSpotHologram(
        spot_vectors=spot_zernike,
        basis=[2, 1, 4],
        cameraslm=z_fs,
    )
    compressed_spot_hologram.optimize(
        method="WGS-Leonardo",
        maxiter=spot_iterations,
        verbose=False,
    )

    _, compressed_figures = collect_figures(
        lambda: (
            plot_spot_map(
                z_cam.shape,
                _spot_ij,
                focus_coefficients,
                "Camera-space spots colored by focus coefficient",
                "ANSI 4 coefficient",
            ),
            plot_coefficient_matrix(
                spot_zernike,
                ["Z2 x tilt", "Z1 y tilt", "Z4 focus"],
                "Compressed spot coefficients",
            ),
            compressed_spot_hologram.plot_nearfield(
                title=f"WGS-Leonardo after {spot_iterations} iterations",
                cbar=True,
            ),
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
                Optimized `{spot_count}` compressed spots with basis `[2, 1, 4]`.
                The hologram shape is `{tuple(compressed_spot_hologram.shape)}`, inherited
                from the simulated SLM.
                """
            ),
            render_figures(compressed_figures),
        ],
        gap=1.0,
    )
    return compressed_spot_hologram


@app.cell
def _(mo):
    mo.md(r"""
    ## Higher-Order Zernike Spot Basis

    The compressed basis can include higher-order aberrations. The first two basis entries
    must still be ANSI 2 and 1 so that lateral spot positions are defined.
    """)
    return


@app.cell
def _(mo):
    basis_controls = mo.md(
        """
        **Higher-order basis controls**

        Pyramid side length: {basis_side}

        Aberration span: {aberration_span}

        WGS iterations: {basis_iterations}
        """
    ).batch(
        basis_side=mo.ui.slider(2, 5, step=1, value=3, show_value=True),
        aberration_span=mo.ui.slider(0.0, 3.0, step=0.5, value=1.5, show_value=True),
        basis_iterations=mo.ui.slider(1, 15, step=1, value=5, show_value=True),
    )
    basis_run = mo.ui.run_button(label="Run higher-order compressed basis")

    mo.vstack([basis_controls, basis_run], gap=1.0)
    return basis_controls, basis_run


@app.cell
def _(basis_controls):
    aberration_span = float(basis_controls.value["aberration_span"])
    basis_iterations = int(basis_controls.value["basis_iterations"])
    basis_side = int(basis_controls.value["basis_side"])
    return aberration_span, basis_iterations, basis_side


@app.cell
def _(
    CompressedSpotHologram,
    aberration_span,
    basis_iterations,
    basis_run,
    basis_side,
    collect_figures,
    mo,
    np,
    plot_coefficient_matrix,
    plot_spot_map,
    render_figures,
    toolbox,
    z_cam,
    z_fs,
):
    mo.stop(
        not basis_run.value,
        mo.md("Click **Run higher-order compressed basis** to compute this section."),
    )

    grid_axis = np.linspace(-1, 1, basis_side)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    _basis_spot_ij = np.vstack(
        (
            z_cam.shape[1] / 2 + 34 * grid_x.ravel(),
            z_cam.shape[0] / 2 + 34 * grid_y.ravel(),
        )
    )
    _basis_lateral_zernike = toolbox.convert_vector(
        _basis_spot_ij,
        from_units="ij",
        to_units="zernike",
        hardware=z_fs,
    )
    sample = np.linspace(-1, 1, _basis_spot_ij.shape[1])
    higher_order_basis = [2, 1, 4, 3, 5, 6]
    higher_order_vectors = np.vstack(
        (
            _basis_lateral_zernike,
            aberration_span * sample,
            aberration_span * np.sin(np.pi * sample),
            aberration_span * np.cos(np.pi * sample),
            0.5 * aberration_span * sample**2,
        )
    )

    basis_hologram = CompressedSpotHologram(
        spot_vectors=higher_order_vectors,
        basis=higher_order_basis,
        cameraslm=z_fs,
    )
    basis_hologram.optimize(
        method="WGS-Leonardo",
        maxiter=basis_iterations,
        verbose=False,
    )

    _, basis_figures = collect_figures(
        lambda: (
            plot_spot_map(
                z_cam.shape,
                _basis_spot_ij,
                higher_order_vectors[2],
                "Camera-space grid colored by focus",
                "ANSI 4 coefficient",
            ),
            plot_coefficient_matrix(
                higher_order_vectors,
                [f"Z{index}" for index in higher_order_basis],
                "Higher-order compressed basis coefficients",
            ),
            basis_hologram.plot_nearfield(
                title=f"Higher-order basis after {basis_iterations} iterations",
                cbar=True,
            ),
        )
    )

    mo.vstack(
        [
            mo.md(
                f"""
                Optimized `{_basis_spot_ij.shape[1]}` spots with basis `{higher_order_basis}`.
                Higher-order coefficients are synthetic here, but they exercise the same
                basis machinery used for calibrated aberration correction.
                """
            ),
            render_figures(basis_figures),
        ],
        gap=1.0,
    )
    return basis_hologram


@app.cell
def _(mo):
    mo.md(r"""
    ## What to Take Away

    - ANSI Zernike indices are the compact labels used throughout `slmsuite`.
    - Derivatives and weighted sums share the same Zernike helper APIs.
    - `CompressedSpotHologram` targets only the requested spots, so plotting the spot map
      and coefficient matrix is more informative than treating its target as a 2D image.
    - The compressed basis can mix tilt, focus, and higher-order aberration coefficients.
    """)
    return


if __name__ == "__main__":
    app.run()
