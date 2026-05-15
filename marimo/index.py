import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import html

    import marimo as mo

    return html, mo


@app.cell
def _(html, mo):
    notebooks = [
        {
            "title": "Computational Holography",
            "route": "/computational",
            "role": "Start here",
            "scope": "Single-pixel targets, GS retrieval, diffraction sampling, spot arrays, WGS, and image targets.",
        },
        {
            "title": "Structured Light",
            "route": "/structured",
            "role": "Phase grammar",
            "scope": "Blazes, lenses, transformed coordinates, LG/HG modes, local imprints, and Zernikes.",
        },
        {
            "title": "Experimental Holography",
            "route": "/experimental",
            "role": "Camera-SLM reality",
            "scope": "Simulated hardware, Fourier calibration, calibrated blazes, feedback holography, and WGS workflows.",
        },
        {
            "title": "Wavefront Calibration",
            "route": "/wavefront",
            "role": "Single-point correction",
            "scope": "Aberrated sources, Fourier calibration prerequisites, superpixel correction, and recalibration.",
        },
        {
            "title": "Zernike Holography",
            "route": "/zernike",
            "role": "Modal basis",
            "scope": "Zernike indexing, phase sums, defocus terms, and compressed spot holograms in Zernike bases.",
        },
        {
            "title": "Multipoint Calibration",
            "route": "/multipoint",
            "role": "Field-dependent correction",
            "scope": "Multipoint superpixel schedules, point spacing, Zernike calibration, and smoothing.",
        },
        {
            "title": "Multiplane Holography",
            "route": "/multiplane",
            "role": "Capstone",
            "scope": "Image planes, compressed spot objectives, propagation kernels, and shared-phase optimization.",
        },
    ]

    learning_path = [
        {
            "number": "01",
            "title": "What an SLM Solves",
            "theory": "A phase-only SLM does not paint brightness. It delays a coherent wavefront so later propagation makes constructive and destructive interference.",
            "equation": "I(kx, ky) proportional to |FFT(A(x, y) * exp(i * phi(x, y)))|^2",
            "open": [("Computational Holography", "/computational")],
            "watch": "The requested single farfield pixel becomes a global nearfield phase pattern; every SLM pixel participates.",
        },
        {
            "number": "02",
            "title": "Fourier Optics and Phase Primitives",
            "theory": "A lens maps SLM-plane structure into spatial-frequency or angle space. Linear phase ramps steer spots; quadratic phase terms move focus.",
            "equation": "phase ramp -> shifted farfield spot; wrapped phase -> physical 2 pi command image",
            "open": [
                ("Structured Light", "/structured"),
                ("Computational Holography", "/computational"),
            ],
            "watch": "Blazes, lenses, coordinate transforms, and padding are the vocabulary used by the later hologram optimizers.",
        },
        {
            "number": "03",
            "title": "Phase Retrieval",
            "theory": "The inverse problem is nonlinear because the SLM controls only phase while the target specifies output intensity. GS alternates between known nearfield amplitude and desired farfield amplitude.",
            "equation": "target amplitude = sqrt(target intensity); unknown target phase is an optimization degree of freedom",
            "open": [("Computational Holography", "/computational")],
            "watch": "GS, WGS, and image targets show why spot weights and unused noise regions matter.",
        },
        {
            "number": "04",
            "title": "Coordinates and Calibration",
            "theory": "The FFT grid, normalized k-space, spatial frequency, angle, and camera pixels are related but not interchangeable. Calibration is the dictionary.",
            "equation": "camera pixels = affine mapping of calibrated Fourier coordinates",
            "open": [("Experimental Holography", "/experimental")],
            "watch": "Basis names such as knm, kxy, and ij are not cosmetic; they determine what a target coordinate means.",
        },
        {
            "number": "05",
            "title": "Real Hardware Feedback",
            "theory": "Finite pixels, phase nonlinearity, illumination rolloff, leakage, camera noise, and alignment error break the ideal FFT model. Feedback measures what actually happened.",
            "equation": "weights <- weights * measured-error correction",
            "open": [
                ("Experimental Holography", "/experimental"),
                ("Wavefront Calibration", "/wavefront"),
            ],
            "watch": "Compare computational feedback with simulated camera feedback; the same intent is corrected in different coordinate systems.",
        },
        {
            "number": "06",
            "title": "Aberrations and Zernikes",
            "theory": "Real optics add wavefront error W(x, y). If the error is measured, the SLM can display approximately -W(x, y). Zernikes provide a compact modal language.",
            "equation": "corrected field = ideal field * exp(i W) * exp(-i W)",
            "open": [
                ("Structured Light", "/structured"),
                ("Wavefront Calibration", "/wavefront"),
                ("Zernike Holography", "/zernike"),
            ],
            "watch": "Zernike indexing conventions matter; the notebook uses the same software basis that later compressed holograms rely on.",
        },
        {
            "number": "07",
            "title": "Field-Dependent Calibration",
            "theory": "A single correction point is not enough when aberrations vary across the field. Multipoint calibration samples several locations and smooths a correction model.",
            "equation": "sparse measured corrections -> Zernike-smoothed field correction",
            "open": [("Multipoint Calibration", "/multipoint")],
            "watch": "The calibration schedule, point spacing, and smoothing choices are part of the optical model.",
        },
        {
            "number": "08",
            "title": "3D and Multiplane Synthesis",
            "theory": "The capstone problem asks one phase-only SLM image to satisfy several objectives: image planes, sparse spots, axial offsets, or modal bases.",
            "equation": "one shared phase mask -> multiple propagated objectives",
            "open": [
                ("Zernike Holography", "/zernike"),
                ("Multiplane Holography", "/multiplane"),
            ],
            "watch": "Multiplane optimization combines the earlier ideas: phase primitives, GS/WGS, calibration, compressed spot bases, and propagation kernels.",
        },
    ]

    translations = [
        ("phi", "SLM-plane phase variable"),
        ("amp", "nearfield illumination amplitude A(x, y)"),
        ("target", "desired farfield amplitude, often sqrt(intensity)"),
        ("weights", "adjustable target amplitudes used by WGS feedback"),
        ("farfield", "FFT-propagated complex field"),
        ('basis="knm"', "computational FFT-grid coordinates"),
        ('basis="kxy"', "calibrated Fourier or normalized k-space coordinates"),
        ('basis="ij"', "camera image pixel coordinates"),
        ("fourier_calibrate()", "fit between SLM Fourier coordinates and camera pixels"),
        ("wavefront_calibrate()", "measure phase error and build a correction map"),
    ]

    audit_notes = [
        "The primer's Fourier shift theorem section has a likely LaTeX typo where \\tilde became a tab plus 'ilde'. Fix that before publishing the primer itself.",
        "Scale factors, FFT normalization, and quadratic phase factors are intentionally suppressed in the primer. The site labels those sections as conceptual rather than exact propagation derivations.",
        "The scalar and paraxial model is the default learning model. High-NA vector effects, polarization-dependent SLM behavior, and device LUT details are advanced corrections.",
        "Weighted GS is a family of update rules in slmsuite, not a single canonical formula.",
        "Zernike indexing must be treated as a software convention, not just a mathematical name.",
    ]

    def esc(value):
        return html.escape(str(value), quote=True)

    def notebook_link(title, route):
        return f'<a class="route-chip" href="{esc(route)}">{esc(title)}</a>'

    notebook_cards = "\n".join(
        f"""
        <a class="notebook-card" href="{esc(notebook["route"])}">
          <span class="notebook-role">{esc(notebook["role"])}</span>
          <span class="notebook-title">{esc(notebook["title"])}</span>
          <span class="notebook-route">{esc(notebook["route"])}</span>
          <span class="notebook-scope">{esc(notebook["scope"])}</span>
        </a>
        """
        for notebook in notebooks
    )

    path_cards = "\n".join(
        f"""
        <section class="path-card">
          <div class="path-number">{esc(module["number"])}</div>
          <div class="path-body">
            <h3>{esc(module["title"])}</h3>
            <p>{esc(module["theory"])}</p>
            <code>{esc(module["equation"])}</code>
            <div class="open-row">
              {"".join(notebook_link(title, route) for title, route in module["open"])}
            </div>
            <p class="watch"><strong>Watch for:</strong> {esc(module["watch"])}</p>
          </div>
        </section>
        """
        for module in learning_path
    )

    translation_cards = "\n".join(
        f"""
        <div class="translation-card">
          <code>{esc(term)}</code>
          <span>{esc(meaning)}</span>
        </div>
        """
        for term, meaning in translations
    )

    audit_items = "\n".join(f"<li>{esc(note)}</li>" for note in audit_notes)

    mo.Html(
        f"""
        <style>
          .slm-site {{
            --text: #111827;
            --muted: #4b5563;
            --line: #d1d5db;
            --panel: #ffffff;
            --soft: #f7f9fb;
            --blue: #2563eb;
            --green: #047857;
            --amber: #b45309;
            max-width: 1180px;
            margin: 0 auto;
            padding: 28px 20px 48px;
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .slm-site * {{
            box-sizing: border-box;
          }}
          .slm-site a {{
            color: inherit;
          }}
          .hero {{
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr);
            gap: 20px;
            align-items: stretch;
            padding: 26px 0 22px;
            border-bottom: 1px solid var(--line);
          }}
          .hero h1 {{
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.12;
            font-weight: 760;
            letter-spacing: 0;
          }}
          .hero p {{
            max-width: 760px;
            margin: 14px 0 0;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
          }}
          .source-panel {{
            padding: 16px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--soft);
          }}
          .source-panel h2 {{
            margin: 0 0 10px;
            font-size: 1rem;
            line-height: 1.3;
          }}
          .source-panel p {{
            margin: 0 0 10px;
            font-size: 0.92rem;
          }}
          .source-panel > code {{
            display: block;
            overflow-wrap: anywhere;
            padding: 8px;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #ffffff;
            color: #1f2937;
            font-size: 0.82rem;
          }}
          .source-panel p code {{
            display: inline;
            padding: 1px 4px;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            background: #ffffff;
            color: #1f2937;
            font-size: 0.86em;
          }}
          .section-heading {{
            margin: 34px 0 14px;
          }}
          .section-heading h2 {{
            margin: 0;
            font-size: 1.35rem;
            line-height: 1.25;
            letter-spacing: 0;
          }}
          .section-heading p {{
            max-width: 780px;
            margin: 8px 0 0;
            color: var(--muted);
            line-height: 1.5;
          }}
          .notebook-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
          }}
          .notebook-card {{
            display: flex;
            min-height: 164px;
            flex-direction: column;
            gap: 8px;
            padding: 16px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            text-decoration: none;
            transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
          }}
          .notebook-card:hover {{
            border-color: var(--blue);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
            transform: translateY(-1px);
          }}
          .notebook-role {{
            width: fit-content;
            padding: 2px 7px;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            color: #1d4ed8;
            background: #eff6ff;
            font-size: 0.78rem;
            font-weight: 650;
          }}
          .notebook-title {{
            font-size: 1.02rem;
            font-weight: 730;
            line-height: 1.25;
          }}
          .notebook-route {{
            width: fit-content;
            color: #065f46;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.83rem;
          }}
          .notebook-scope {{
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.42;
          }}
          .path-stack {{
            display: grid;
            gap: 12px;
          }}
          .path-card {{
            display: grid;
            grid-template-columns: 62px minmax(0, 1fr);
            gap: 14px;
            padding: 16px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
          }}
          .path-number {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 46px;
            height: 46px;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            background: #f0fdf4;
            color: var(--green);
            font-weight: 760;
          }}
          .path-body h3 {{
            margin: 0;
            font-size: 1.05rem;
            line-height: 1.3;
          }}
          .path-body p {{
            margin: 8px 0 0;
            color: var(--muted);
            line-height: 1.48;
          }}
          .path-body code {{
            display: block;
            width: fit-content;
            max-width: 100%;
            margin-top: 10px;
            padding: 7px 9px;
            overflow-wrap: anywhere;
            border: 1px solid #fde68a;
            border-radius: 6px;
            background: #fffbeb;
            color: #78350f;
            font-size: 0.86rem;
          }}
          .open-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
          }}
          .route-chip {{
            display: inline-flex;
            align-items: center;
            min-height: 30px;
            padding: 5px 9px;
            border: 1px solid #c7d2fe;
            border-radius: 6px;
            background: #eef2ff;
            color: #3730a3;
            text-decoration: none;
            font-size: 0.86rem;
            font-weight: 650;
          }}
          .route-chip:hover {{
            border-color: var(--blue);
            background: #dbeafe;
          }}
          .watch {{
            font-size: 0.92rem;
          }}
          .translation-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 10px;
          }}
          .translation-card {{
            display: grid;
            grid-template-columns: minmax(98px, 0.75fr) minmax(0, 1.25fr);
            gap: 10px;
            align-items: start;
            padding: 12px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--soft);
          }}
          .translation-card code {{
            overflow-wrap: anywhere;
            color: #0f766e;
            font-size: 0.84rem;
          }}
          .translation-card span {{
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.4;
          }}
          .audit-box {{
            margin-top: 24px;
            padding: 14px 16px;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            background: #fff7ed;
            color: #7c2d12;
          }}
          .audit-box summary {{
            cursor: pointer;
            font-weight: 720;
          }}
          .audit-box ul {{
            margin: 10px 0 0;
            padding-left: 18px;
          }}
          .audit-box li {{
            margin: 6px 0;
            line-height: 1.45;
          }}
          @media (max-width: 760px) {{
            .slm-site {{
              padding: 18px 12px 38px;
            }}
            .hero {{
              grid-template-columns: 1fr;
              padding-top: 12px;
            }}
            .hero h1 {{
              font-size: 1.75rem;
            }}
            .path-card {{
              grid-template-columns: 1fr;
            }}
            .translation-card {{
              grid-template-columns: 1fr;
            }}
          }}
          @media (prefers-color-scheme: dark) {{
            .slm-site {{
              --text: #f9fafb;
              --muted: #d1d5db;
              --line: #374151;
              --panel: #111827;
              --soft: #0f172a;
            }}
            .source-panel > code,
            .source-panel p code {{
              border-color: #374151;
              background: #020617;
              color: #e5e7eb;
            }}
            .notebook-role {{
              border-color: #1d4ed8;
              background: #172554;
              color: #bfdbfe;
            }}
            .notebook-route {{
              color: #a7f3d0;
            }}
            .path-number {{
              border-color: #047857;
              background: #052e2b;
              color: #6ee7b7;
            }}
            .path-body code {{
              border-color: #92400e;
              background: #451a03;
              color: #fde68a;
            }}
            .route-chip {{
              border-color: #3730a3;
              background: #1e1b4b;
              color: #c7d2fe;
            }}
            .route-chip:hover {{
              background: #312e81;
            }}
            .translation-card code {{
              color: #5eead4;
            }}
            .audit-box {{
              border-color: #9a3412;
              background: #431407;
              color: #fed7aa;
            }}
          }}
        </style>
        <main class="slm-site">
          <section class="hero">
            <div>
              <h1>slmsuite Learning Site</h1>
              <p>
                This entry point turns the Marimo examples into a guided path through
                phase-only holography: first the wave and Fourier theory, then the
                slmsuite objects, coordinate bases, calibration steps, and advanced
                optimization workflows.
              </p>
            </div>
            <aside class="source-panel">
              <h2>Theory source</h2>
              <p>
                Synthesized from the local SLM theory primer and the seven interactive
                notebooks now kept under the repo's <code>marimo/</code> directory.
              </p>
              <code>/Users/aadarwal/Downloads/slm_theory_primer.tex</code>
            </aside>
          </section>

          <section class="section-heading">
            <h2>Open a Notebook</h2>
            <p>
              These are the current interactive notebooks. The roles make the dependency
              chain explicit, but each card still opens directly.
            </p>
          </section>
          <section class="notebook-grid" aria-label="Marimo notebooks">
            {notebook_cards}
          </section>

          <section class="section-heading">
            <h2>Learning Path</h2>
            <p>
              Each step names the active theory, the slmsuite notebook that makes it
              concrete, and the main thing to inspect when you run the cells.
            </p>
          </section>
          <section class="path-stack" aria-label="Theory guided learning path">
            {path_cards}
          </section>

          <section class="section-heading">
            <h2>Theory to Code</h2>
            <p>
              Keep this translation nearby while reading the notebooks. Most confusion
              comes from mixing optical fields, target amplitudes, FFT indices, calibrated
              Fourier units, and camera pixels.
            </p>
          </section>
          <section class="translation-grid" aria-label="Theory to slmsuite translation">
            {translation_cards}
          </section>

          <details class="audit-box">
            <summary>Primer audit notes to carry into the full synthesis</summary>
            <ul>
              {audit_items}
            </ul>
          </details>
        </main>
        """
    )
    return audit_notes, learning_path, notebooks, translations


if __name__ == "__main__":
    app.run()
