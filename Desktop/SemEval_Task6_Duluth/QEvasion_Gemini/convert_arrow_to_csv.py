import pyarrow as pa
import pyarrow.csv as pcsv
import os

base = os.path.dirname(os.path.abspath(__file__))

for split in ["train", "test"]:
    arrow_path = os.path.join(base, split, "data-00000-of-00001.arrow")
    csv_path = os.path.join(base, split, "data.csv")
    with open(arrow_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    pcsv.write_csv(table, csv_path)
    print(f"Saved {csv_path} ({table.num_rows} rows)")
