"""Template: fit models on your own spectra.

Export your data as a wide CSV (one row per spectrum)::

    sample_id, batch, day, glucose, lactate, 400, 401, 402, ..., 1800
    S01,       R1,    0,   5.78,    0.07,    812, 815, 820, ...

Column headers that are numbers are read as wavenumbers; all other columns
become metadata. From MATLAB you can instead save a .mat file::

    save('spectra.mat', 'X', 'wavenumbers', 'glucose', 'lactate', 'batch', '-v7')

and use ``load_mat`` below.

Run with:  python examples/03_your_own_data.py path/to/data.csv
"""

import sys

from specpls import PLSModel, load_csv

if len(sys.argv) < 2:
    sys.exit(__doc__)

data = load_csv(sys.argv[1])
# data = load_mat("spectra.mat", meta_keys=["glucose", "lactate", "batch"])

for analyte in ["glucose", "lactate"]:
    subset = data.dropna(analyte)
    model = PLSModel(
        preprocessing="savgol:15:2:1,snv",
        regions=[(400, 1800)],
        max_components=10,
        cv="batch",  # needs a 'batch' column; use cv=10 for venetian blinds
    ).fit(
        subset.X,
        subset.target(analyte),
        wavenumbers=subset.wavenumbers,
        groups=subset.meta["batch"],
        target_name=analyte,
    )
    print(model.summary())
    model.save(f"{analyte}_model.joblib")
