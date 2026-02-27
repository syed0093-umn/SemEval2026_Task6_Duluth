"""
Convert QEvasion_Gemini CSV files to Arrow format for HuggingFace datasets.
This creates the data-00000-of-00001.arrow file needed by load_from_disk().
"""

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def convert_split(split_dir, split_name):
    csv_path = os.path.join(split_dir, 'data.csv')
    arrow_path = os.path.join(split_dir, 'data-00000-of-00001.arrow')

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Ensure correct dtypes to match dataset_info.json schema
    str_cols = ['title', 'date', 'president', 'url', 'interview_question',
                'interview_answer', 'gpt3.5_summary', 'gpt3.5_prediction',
                'question', 'annotator_id', 'annotator1', 'annotator2',
                'annotator3', 'clarity_label', 'evasion_label']
    int_cols = ['question_order', 'index']
    bool_cols = ['inaudible', 'multiple_questions', 'affirmative_questions']

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype('int64')
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Convert to PyArrow Table
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Write as Arrow IPC (streaming format, same as HuggingFace datasets)
    print(f"  Writing {arrow_path}...")
    with pa.OSFile(arrow_path, 'wb') as f:
        writer = ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()

    arrow_size = os.path.getsize(arrow_path)
    print(f"  Arrow file: {arrow_size:,} bytes")

    # Update dataset_info.json
    info_path = os.path.join(split_dir, '..', 'dataset_info.json') if os.path.exists(
        os.path.join(split_dir, '..', 'dataset_info.json')) else None

    return len(df), arrow_size


def update_dataset_info(gemini_dir, train_count, train_bytes, test_count, test_bytes):
    info_path = os.path.join(gemini_dir, 'train', 'dataset_info.json')
    if not os.path.exists(info_path):
        print(f"  No dataset_info.json found, skipping update")
        return

    with open(info_path, 'r') as f:
        info = json.load(f)

    info['splits']['train']['num_examples'] = train_count
    info['splits']['train']['num_bytes'] = train_bytes
    info['splits']['test']['num_examples'] = test_count
    info['splits']['test']['num_bytes'] = test_bytes
    info['dataset_size'] = train_bytes + test_bytes

    # Write updated info to both train and test dirs
    for subdir in ['train', 'test']:
        out_path = os.path.join(gemini_dir, subdir, 'dataset_info.json')
        with open(out_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"  Updated {out_path}")


if __name__ == '__main__':
    gemini_dir = os.path.join(BASE_DIR, 'QEvasion_Gemini')

    print("=" * 60)
    print("Converting QEvasion_Gemini CSV to Arrow format")
    print("=" * 60)

    train_dir = os.path.join(gemini_dir, 'train')
    test_dir = os.path.join(gemini_dir, 'test')

    train_count, train_bytes = convert_split(train_dir, 'train')
    test_count, test_bytes = convert_split(test_dir, 'test')

    update_dataset_info(gemini_dir, train_count, train_bytes, test_count, test_bytes)

    print("\nDone! Verifying with load_from_disk...")
    from datasets import load_from_disk
    ds = load_from_disk(gemini_dir)
    print(f"  Train: {len(ds['train'])} examples")
    print(f"  Test: {len(ds['test'])} examples")
    print(f"  Columns: {ds['train'].column_names}")
    print("Conversion successful!")
