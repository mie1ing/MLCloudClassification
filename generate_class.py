import os
import pandas as pd

top_dir = "GCD/test"
sub_dirs = os.listdir(top_dir)

for sub_dir in sub_dirs:
    if not os.path.isdir(os.path.join(top_dir, sub_dir)):
        continue
    cls = sub_dir
    files = [f for f in os.listdir(os.path.join(top_dir, sub_dir))]
    if not files:
        continue
    record = []
    for f in files:
        record.append((f, cls))

    if record:
        pd.DataFrame(record, columns=["filename", "class"]).to_csv(
            f"{top_dir}/{sub_dir}/{cls}.csv", index=False
        )