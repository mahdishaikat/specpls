"""Compare preprocessing strategies for several analytes in one table.

Every combination is evaluated with leave-one-batch-out cross-validation on
the calibration runs and on an independent hold-out run, giving a table in
the format commonly used in publications (LVs, RMSEC, RMSECV, RMSEP, R2).

Run with:  python examples/02_compare_preprocessing.py
"""

import pandas as pd

from specpls import PLSModel
from specpls.datasets import make_synthetic_cell_culture

PREPROCESSING = {
    "raw": None,
    "SNV": "snv",
    "MSC": "msc",
    "SG(11,2,1st)": "savgol:11:2:1",
    "SG(11,2,1st)+SNV": "savgol:11:2:1,snv",
    "AsLS+SNV": "asls:1e5:0.01,snv",
}
ANALYTES = ["glucose", "lactate"]

data = make_synthetic_cell_culture(random_state=1)
is_test = (data.meta["batch"] == "B6").to_numpy()
cal, test = data.subset(~is_test), data.subset(is_test)

rows = []
for analyte in ANALYTES:
    for label, spec in PREPROCESSING.items():
        model = PLSModel(preprocessing=spec, max_components=10, cv="batch").fit(
            cal.X,
            cal.target(analyte),
            groups=cal.meta["batch"],
            target_name=analyte,
        )
        row = model.summary(test.X, test.target(analyte))
        row["preprocessing"] = label
        rows.append(row)

table = pd.DataFrame(rows).set_index(["target", "preprocessing"]).round(3)
print(table.to_string())

best = table.groupby(level="target")["rmsecv"].idxmin()
print("\nLowest RMSECV per analyte (choose on CV, not on the test set):")
for analyte, (_, label) in best.items():
    print(f"  {analyte}: {label}")
