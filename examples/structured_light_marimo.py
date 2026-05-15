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
    # Structured Light with `slmsuite`

    This marimo notebook mirrors the structured-light documentation example with a
    simulated SLM/camera pair. The controls expose the phase-building helpers used for
    blazes, lenses, transformed coordinate bases, analytic spatial modes, segmented
    imprints, and Zernike aberrations.
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

    def plot_phase_series(slm, entries):
        axes = []
        for title, phase_data in entries:
            plt.figure(figsize=(8, 4))
            axes.append(slm.plot(phase_data, title=title))
        return axes

    def plot_array_series(entries, cmap="viridis"):
        axes = []
        for title, data in entries:
            fig, ax = plt.subplots(figsize=(6, 4))
            image = ax.imshow(data, interpolation="none", cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            axes.append(ax)
        return axes

    def wrap_phase(phase_data):
        return np.mod(phase_data, 2 * np.pi)

    return collect_figures, plot_array_series, plot_phase_series, render_figures, wrap_phase


@app.cell
def _(mo):
    setup_controls = mo.md(
        """
        **Simulated hardware**

        SLM width: {slm_width}

        Camera width: {camera_width}

        Camera rotation: {rotation}

        Gaussian source width: {source_width}
        """
    ).batch(
        slm_width=mo.ui.slider(192, 512, step=64, value=256, show_value=True),
        camera_width=mo.ui.slider(192, 512, step=64, value=320, show_value=True),
        rotation=mo.ui.slider(-8.0, 8.0, step=1.0, value=4.0, show_value=True),
        source_width=mo.ui.slider(0.25, 0.60, step=0.05, value=0.40, show_value=True),
    )
    setup_run = mo.ui.run_button(label="Rebuild simulated hardware")

    mo.vstack([setup_controls, setup_run], gap=1.0)
    return setup_controls, setup_run


@app.cell
def _(setup_controls):
    camera_width = int(setup_controls.value["camera_width"])
    rotation_deg = float(setup_controls.value["rotation"])
    slm_width = int(setup_controls.value["slm_width"])
    source_width = float(setup_controls.value["source_width"])
    return camera_width, rotation_deg, slm_width, source_width


@app.cell
def _(
    FourierSLM,
    SimulatedCamera,
    SimulatedSLM,
    camera_width,
    collect_figures,
    np,
    plot_array_series,
    rotation_deg,
    setup_run,
    slm_width,
    source_width,
    warnings,
):
    setup_run.value

    slm_resolution = (slm_width, int(round(slm_width * 0.75)))
    camera_resolution = (camera_width, int(round(camera_width * 0.76)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"'camera' _get_image_hw\(\) failed .*")
        struct_slm = SimulatedSLM(slm_resolution, pitch_um=(8, 8))
        source_kwargs = dict(
            units="frac",
            x0=0,
            y0=0,
            a=1,
            c=0,
            wx=source_width,
            wy=source_width * 0.85,
        )
        struct_slm.set_source_analytic(**source_kwargs, sim=True)
        struct_slm.set_source_analytic(**source_kwargs)
        struct_cam = SimulatedCamera(
            struct_slm,
            camera_resolution,
            pitch_um=(4, 4),
            gain=1,
        )

    struct_fs = FourierSLM(struct_cam, struct_slm)
    struct_M, struct_b = struct_fs.fourier_calibration_build(
        f_eff=80000.0,
        theta=rotation_deg * np.pi / 180,
    )
    struct_cam.set_affine(struct_M, struct_b)
    struct_fs.fourier_calibrate_analytic(struct_M, struct_b)
    struct_cam.set_exposure(0.03)

    _, source_figures = collect_figures(
        lambda: plot_array_series(
            [
                ("SLM source amplitude", struct_slm.source["amplitude"]),
                ("Simulated source amplitude", struct_slm.source["amplitude_sim"]),
            ]
        )
    )
    return (
        source_figures,
        struct_b,
        struct_cam,
        struct_fs,
        struct_M,
        struct_slm,
    )


@app.cell
def _(mo, np, render_figures, source_figures, struct_b, struct_cam, struct_M, struct_slm):
    mo.vstack(
        [
            mo.md(
                f"""
                Active simulation:

                - SLM shape: `{struct_slm.shape}` pixels
                - Camera shape: `{struct_cam.shape}` pixels
                - Fourier matrix determinant: `{np.linalg.det(struct_M):.3g}`
                - Camera offset: `{np.squeeze(struct_b).round(2).tolist()}`
                """
            ),
            render_figures(source_figures, columns=2),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Blazes and Lenses

    The basic structured-light primitives are linear blazes and quadratic lens terms.
    Their phases can be added directly, then wrapped onto the SLM's available phase range.
    """)
    return


@app.cell
def _(mo):
    blaze_lens_controls = mo.md(
        """
        **Blaze/lens controls**

        kx in mrad: {kx_mrad}

        ky in mrad: {ky_mrad}

        Spherical lens scale: {lens_scale}

        Ellipticity: {ellipticity}
        """
    ).batch(
        kx_mrad=mo.ui.slider(-3.0, 3.0, step=0.25, value=1.0, show_value=True),
        ky_mrad=mo.ui.slider(-3.0, 3.0, step=0.25, value=-0.75, show_value=True),
        lens_scale=mo.ui.slider(0.5, 4.0, step=0.25, value=1.5, show_value=True),
        ellipticity=mo.ui.slider(0.25, 2.0, step=0.25, value=1.0, show_value=True),
    )
    blaze_lens_run = mo.ui.run_button(label="Render blaze/lens phases")

    mo.vstack([blaze_lens_controls, blaze_lens_run], gap=1.0)
    return blaze_lens_controls, blaze_lens_run


@app.cell
def _(blaze_lens_controls):
    ellipticity = float(blaze_lens_controls.value["ellipticity"])
    kx_mrad = float(blaze_lens_controls.value["kx_mrad"])
    ky_mrad = float(blaze_lens_controls.value["ky_mrad"])
    lens_scale = float(blaze_lens_controls.value["lens_scale"])
    return ellipticity, kx_mrad, ky_mrad, lens_scale


@app.cell
def _(
    blaze_lens_run,
    collect_figures,
    ellipticity,
    kx_mrad,
    ky_mrad,
    lens_scale,
    phase_tools,
    plot_phase_series,
    struct_slm,
):
    blaze_lens_run.value

    blaze_phase = phase_tools.blaze(struct_slm, vector=(kx_mrad * 1e-3, ky_mrad * 1e-3))
    spherical_phase = phase_tools.lens(struct_slm, f=1e7 / lens_scale)
    elliptical_phase = phase_tools.lens(
        struct_slm,
        f=(1e7 / lens_scale, 1e7 / (lens_scale * ellipticity)),
    )
    combined_phase = blaze_phase + elliptical_phase

    _, blaze_lens_figures = collect_figures(
        lambda: plot_phase_series(
            struct_slm,
            [
                ("Linear blaze", blaze_phase),
                ("Spherical lens", spherical_phase),
                ("Elliptical lens", elliptical_phase),
                ("Blaze + elliptical lens", combined_phase),
            ],
        )
    )
    return blaze_lens_figures


@app.cell
def _(blaze_lens_figures, render_figures):
    render_figures(blaze_lens_figures, columns=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Coordinate Transforms

    `toolbox.transform_grid` rotates and shifts the coordinate basis before a phase function
    is evaluated. This is often cleaner than generating a phase and then resampling it.
    """)
    return


@app.cell
def _(mo):
    transform_controls = mo.md(
        """
        **Transformed-basis controls**

        Basis rotation: {basis_rotation}

        x shift: {x_shift}

        y shift: {y_shift}
        """
    ).batch(
        basis_rotation=mo.ui.slider(-60.0, 60.0, step=5.0, value=30.0, show_value=True),
        x_shift=mo.ui.slider(-250.0, 250.0, step=25.0, value=75.0, show_value=True),
        y_shift=mo.ui.slider(-250.0, 250.0, step=25.0, value=-50.0, show_value=True),
    )
    transform_run = mo.ui.run_button(label="Render transformed basis")

    mo.vstack([transform_controls, transform_run], gap=1.0)
    return transform_controls, transform_run


@app.cell
def _(transform_controls):
    basis_rotation_deg = float(transform_controls.value["basis_rotation"])
    x_shift_norm = float(transform_controls.value["x_shift"])
    y_shift_norm = float(transform_controls.value["y_shift"])
    return basis_rotation_deg, x_shift_norm, y_shift_norm


@app.cell
def _(
    basis_rotation_deg,
    collect_figures,
    np,
    phase_tools,
    plot_phase_series,
    struct_slm,
    toolbox,
    transform_run,
    x_shift_norm,
    y_shift_norm,
):
    transform_run.value

    transformed_grid = toolbox.transform_grid(
        struct_slm,
        transform=basis_rotation_deg * np.pi / 180,
        shift=(x_shift_norm, y_shift_norm),
    )
    normal_lens = phase_tools.lens(struct_slm, f=7e6)
    transformed_lens = phase_tools.lens(transformed_grid, f=(7e6, 2.5e6))

    _, transform_figures = collect_figures(
        lambda: plot_phase_series(
            struct_slm,
            [
                ("Untransformed lens", normal_lens),
                ("Rotated and shifted elliptical lens", transformed_lens),
            ],
        )
    )
    return transform_figures


@app.cell
def _(render_figures, transform_figures):
    render_figures(transform_figures, columns=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Analytic Spatial Modes

    Laguerre-Gaussian and Hermite-Gaussian helpers provide phase masks for common
    structured-light modes. Additional blaze or lens terms can be added the same way
    as in the previous section.
    """)
    return


@app.cell
def _(mo):
    mode_controls = mo.md(
        """
        **Mode controls**

        LG azimuthal order l: {lg_l}

        LG radial order p: {lg_p}

        HG n: {hg_n}

        HG m: {hg_m}

        Added defocus: {mode_defocus}
        """
    ).batch(
        lg_l=mo.ui.slider(-6, 6, step=1, value=3, show_value=True),
        lg_p=mo.ui.slider(0, 4, step=1, value=1, show_value=True),
        hg_n=mo.ui.slider(0, 5, step=1, value=2, show_value=True),
        hg_m=mo.ui.slider(0, 5, step=1, value=1, show_value=True),
        mode_defocus=mo.ui.slider(-2.0, 2.0, step=0.25, value=0.75, show_value=True),
    )
    mode_run = mo.ui.run_button(label="Render spatial modes")

    mo.vstack([mode_controls, mode_run], gap=1.0)
    return mode_controls, mode_run


@app.cell
def _(mode_controls):
    hg_m = int(mode_controls.value["hg_m"])
    hg_n = int(mode_controls.value["hg_n"])
    lg_l = int(mode_controls.value["lg_l"])
    lg_p = int(mode_controls.value["lg_p"])
    mode_defocus = float(mode_controls.value["mode_defocus"])
    return hg_m, hg_n, lg_l, lg_p, mode_defocus


@app.cell
def _(
    collect_figures,
    hg_m,
    hg_n,
    lg_l,
    lg_p,
    mode_defocus,
    mode_run,
    np,
    phase_tools,
    plot_phase_series,
    struct_slm,
):
    mode_run.value

    lg_phase = phase_tools.laguerre_gaussian(struct_slm, l=lg_l, p=lg_p)
    hg_phase = phase_tools.hermite_gaussian(struct_slm, n=hg_n, m=hg_m)
    defocus_phase = phase_tools.lens(struct_slm, f=np.inf if mode_defocus == 0 else 1e7 / mode_defocus)
    tilted_mode = lg_phase + defocus_phase + phase_tools.blaze(struct_slm, vector=(0.0012, -0.0008))

    _, mode_figures = collect_figures(
        lambda: plot_phase_series(
            struct_slm,
            [
                (f"Laguerre-Gaussian l={lg_l}, p={lg_p}", lg_phase),
                (f"Hermite-Gaussian n={hg_n}, m={hg_m}", hg_phase),
                ("LG mode + blaze + defocus", tilted_mode),
            ],
        )
    )
    return mode_figures


@app.cell
def _(mode_figures, render_figures):
    render_figures(mode_figures)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Segmented Imprints

    `toolbox.imprint` writes phase functions into selected windows. The windows can be
    rectangular, circular, or generated from a Voronoi tessellation of control points.
    """)
    return


@app.cell
def _(mo):
    segment_controls = mo.md(
        """
        **Segment controls**

        Segment columns: {segment_cols}

        Segment rows: {segment_rows}

        Voronoi radius fraction: {radius_fraction}

        Local blaze scale: {local_scale}
        """
    ).batch(
        segment_cols=mo.ui.slider(3, 8, step=1, value=5, show_value=True),
        segment_rows=mo.ui.slider(2, 6, step=1, value=4, show_value=True),
        radius_fraction=mo.ui.slider(0.35, 0.80, step=0.05, value=0.55, show_value=True),
        local_scale=mo.ui.slider(0.5, 3.0, step=0.25, value=1.5, show_value=True),
    )
    segment_run = mo.ui.run_button(label="Render segmented imprints")

    mo.vstack([segment_controls, segment_run], gap=1.0)
    return segment_controls, segment_run


@app.cell
def _(segment_controls):
    local_scale = float(segment_controls.value["local_scale"])
    radius_fraction = float(segment_controls.value["radius_fraction"])
    segment_cols = int(segment_controls.value["segment_cols"])
    segment_rows = int(segment_controls.value["segment_rows"])
    return local_scale, radius_fraction, segment_cols, segment_rows


@app.cell
def _(
    collect_figures,
    local_scale,
    np,
    phase_tools,
    plot_phase_series,
    radius_fraction,
    segment_cols,
    segment_rows,
    segment_run,
    struct_slm,
    toolbox,
):
    segment_run.value

    control_points = toolbox.fit_3pt(
        (0.20 * struct_slm.shape[1], 0.22 * struct_slm.shape[0]),
        (0.80 * struct_slm.shape[1], 0.22 * struct_slm.shape[0]),
        (0.20 * struct_slm.shape[1], 0.78 * struct_slm.shape[0]),
        N=(segment_cols, segment_rows),
    )
    nearest_distance = toolbox.smallest_distance(control_points)
    voronoi_radius = radius_fraction * nearest_distance
    segment_phase = np.zeros(struct_slm.shape)

    _, voronoi_figures = collect_figures(
        lambda: toolbox.voronoi_windows(
            struct_slm.shape,
            control_points,
            radius=voronoi_radius,
            plot=True,
        )
    )
    segment_windows = toolbox.voronoi_windows(
        struct_slm.shape,
        control_points,
        radius=voronoi_radius,
        plot=False,
    )

    for index, window in enumerate(segment_windows):
        centered_x = (control_points[0, index] - struct_slm.shape[1] / 2) / struct_slm.shape[1]
        centered_y = (control_points[1, index] - struct_slm.shape[0] / 2) / struct_slm.shape[0]
        toolbox.imprint(
            segment_phase,
            window,
            phase_tools.blaze,
            grid=struct_slm,
            vector=(0.004 * local_scale * centered_x, 0.004 * local_scale * centered_y),
        )

    _, segment_phase_figures = collect_figures(
        lambda: plot_phase_series(
            struct_slm,
            [
                ("Segmented local blazes", segment_phase),
            ],
        )
    )
    return nearest_distance, segment_phase_figures, voronoi_figures


@app.cell
def _(mo, nearest_distance, render_figures, segment_phase_figures, voronoi_figures):
    mo.vstack(
        [
            mo.md(f"Nearest control-point spacing: `{nearest_distance:.2f}` pixels."),
            render_figures(voronoi_figures),
            render_figures(segment_phase_figures),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Zernike Aberrations

    Zernike polynomials give a compact basis for common pupil aberrations. `slmsuite`
    supports individual polynomials, weighted sums, derivatives, and pyramid plots.
    """)
    return


@app.cell
def _(mo):
    zernike_controls = mo.md(
        """
        **Zernike controls**

        ANSI index: {zernike_index}

        Single-term weight: {zernike_weight}

        Pyramid radial order: {pyramid_order}

        Derivative dx: {derivative_x}

        Derivative dy: {derivative_y}
        """
    ).batch(
        zernike_index=mo.ui.slider(0, 18, step=1, value=4, show_value=True),
        zernike_weight=mo.ui.slider(-4.0, 4.0, step=0.5, value=2.0, show_value=True),
        pyramid_order=mo.ui.slider(2, 5, step=1, value=3, show_value=True),
        derivative_x=mo.ui.slider(0, 2, step=1, value=0, show_value=True),
        derivative_y=mo.ui.slider(0, 2, step=1, value=0, show_value=True),
    )
    zernike_run = mo.ui.run_button(label="Render Zernike phases")

    mo.vstack([zernike_controls, zernike_run], gap=1.0)
    return zernike_controls, zernike_run


@app.cell
def _(zernike_controls):
    derivative_x = int(zernike_controls.value["derivative_x"])
    derivative_y = int(zernike_controls.value["derivative_y"])
    pyramid_order = int(zernike_controls.value["pyramid_order"])
    zernike_index = int(zernike_controls.value["zernike_index"])
    zernike_weight = float(zernike_controls.value["zernike_weight"])
    return derivative_x, derivative_y, pyramid_order, zernike_index, zernike_weight


@app.cell
def _(
    collect_figures,
    derivative_x,
    derivative_y,
    phase_tools,
    plot_phase_series,
    pyramid_order,
    struct_slm,
    zernike_index,
    zernike_run,
    zernike_weight,
):
    zernike_run.value

    derivative = (derivative_x, derivative_y)
    single_zernike = phase_tools.zernike(
        struct_slm,
        index=zernike_index,
        weight=zernike_weight,
        derivative=derivative,
        use_mask=False,
    )
    summed_zernike = phase_tools.zernike_sum(
        struct_slm,
        indices=(3, 4, 5, 7, 8),
        weights=(1.0, -1.8, 1.2, 0.8, -0.6),
        derivative=derivative,
        use_mask=False,
    )
    _, zernike_phase_figures = collect_figures(
        lambda: plot_phase_series(
            struct_slm,
            [
                (f"ANSI Zernike {zernike_index}", single_zernike),
                ("Weighted Zernike sum", summed_zernike),
            ],
        )
    )
    _, zernike_pyramid_figures = collect_figures(
        lambda: phase_tools.zernike_pyramid_plot(
            struct_slm,
            order=pyramid_order,
            derivative=derivative,
            use_mask=False,
            scale=max(abs(zernike_weight), 1),
            titles=["ansi", "name"],
        )
    )
    return zernike_phase_figures, zernike_pyramid_figures


@app.cell
def _(render_figures, zernike_phase_figures, zernike_pyramid_figures):
    render_figures(zernike_phase_figures, columns=2)
    render_figures(zernike_pyramid_figures)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What to Take Away

    - Blazes, lenses, analytic modes, and Zernike terms are phase arrays on a common SLM grid.
    - `toolbox.transform_grid` changes the coordinate basis before evaluating a phase helper.
    - `toolbox.imprint` localizes phase functions to rectangular, circular, or Voronoi windows.
    - These primitives can be added before writing the wrapped phase to an SLM.
    """)
    return


if __name__ == "__main__":
    app.run()
