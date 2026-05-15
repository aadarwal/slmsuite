import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    notebook_rows = [
        (
            "Computational Holography",
            "/computational",
            "Single-pixel targets, GS retrieval, diffraction sampling, spot arrays, and image targets.",
        ),
        (
            "Experimental Holography",
            "/experimental",
            "Simulated hardware, Fourier calibration, camera feedback, and weighted GS workflows.",
        ),
        (
            "Wavefront Calibration",
            "/wavefront",
            "Aberrated simulated sources, Fourier calibration, and superpixel correction tests.",
        ),
        (
            "Structured Light",
            "/structured",
            "Blazes, lenses, transformed coordinates, LG/HG modes, imprints, and Zernikes.",
        ),
        (
            "Zernike Holography",
            "/zernike",
            "Zernike indexing, phase sums, and compressed spot holograms in Zernike bases.",
        ),
        (
            "Multipoint Calibration",
            "/multipoint",
            "Multipoint superpixel schedules, calibration-point spacing, and Zernike smoothing.",
        ),
        (
            "Multiplane Holography",
            "/multiplane",
            "Image planes, compressed spot bases, and shared-phase multiplane optimization.",
        ),
    ]

    cards = "\n".join(
        f"""
        <a class="notebook-card" href="{path}">
          <span class="notebook-title">{title}</span>
          <span class="notebook-route">{path}</span>
          <span class="notebook-description">{description}</span>
        </a>
        """
        for title, path, description in notebook_rows
    )

    mo.Html(
        f"""
        <style>
          .slmsuite-index {{
            max-width: 960px;
            margin: 0 auto;
            padding: 24px 0 40px;
            color: #111827;
          }}
          .slmsuite-index h1 {{
            margin: 0 0 8px;
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 700;
            letter-spacing: 0;
          }}
          .slmsuite-index .intro {{
            margin: 0 0 24px;
            color: #4b5563;
            font-size: 0.98rem;
            line-height: 1.5;
          }}
          .notebook-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
          }}
          .notebook-card {{
            display: flex;
            flex-direction: column;
            min-height: 140px;
            padding: 16px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: #ffffff;
            color: inherit;
            text-decoration: none;
            transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
          }}
          .notebook-card:hover {{
            border-color: #2563eb;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
            transform: translateY(-1px);
          }}
          .notebook-title {{
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.3;
          }}
          .notebook-route {{
            width: fit-content;
            margin-top: 8px;
            padding: 2px 7px;
            border-radius: 6px;
            background: #eef2ff;
            color: #3730a3;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.82rem;
          }}
          .notebook-description {{
            margin-top: 12px;
            color: #4b5563;
            font-size: 0.92rem;
            line-height: 1.45;
          }}
          @media (prefers-color-scheme: dark) {{
            .slmsuite-index {{
              color: #f9fafb;
            }}
            .slmsuite-index .intro,
            .notebook-description {{
              color: #d1d5db;
            }}
            .notebook-card {{
              background: #111827;
              border-color: #374151;
            }}
            .notebook-card:hover {{
              border-color: #60a5fa;
              box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
            }}
            .notebook-route {{
              background: #1e3a8a;
              color: #dbeafe;
            }}
          }}
        </style>
        <main class="slmsuite-index">
          <h1>slmsuite Marimo Examples</h1>
          <p class="intro">
            Choose a notebook to open the interactive tutorial.
          </p>
          <section class="notebook-grid" aria-label="Marimo notebooks">
            {cards}
          </section>
        </main>
        """
    )
    return notebook_rows


if __name__ == "__main__":
    app.run()
