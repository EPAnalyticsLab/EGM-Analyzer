import os
import re
from io import StringIO

import pandas as pd

from import_data.decode_raw_content import decode_raw_content
from import_data.extract_coords import extract_coords
from import_data.extract_lessions import extract_lessions

FILE_OF_FILES_RE = re.compile(r"(\d+)\sof\s(\d+)")

REQUIRED_DATA_BLOCKS = ["rov trace"]
REQUIRED_DATA_TABLE_COLUMNS = [
    "pt number",
    "rov LAT",
    "peak2peak",
    "roving x",
    "roving y",
    "roving z",
    "Sample rate",
]
DATA_BLOCKS_TO_READ_COORDS_FROM = ["rov trace"]


def extract_local_dxl_data(paths):
    csv_paths = [path for path in paths if path.endswith(".csv")]

    if len(csv_paths) == 0:
        raise Exception("No DxL files uploaded")

    print(f"Found {len(csv_paths)} potential DxL files")

    def reader(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    names_paths = {os.path.basename(path): path for path in csv_paths}

    return extract(names_paths, reader)


def extract_dxl_data(filenames, contents):
    csv_files = [path for path in filenames if path.endswith(".csv")]

    if len(csv_files) == 0:
        raise Exception("No DxL files uploaded")

    print(f"Found {len(csv_files)} potential DxL files")

    def reader(content):
        return decode_raw_content(content)

    names_contents = {name: content for name, content in zip(filenames, contents)}

    return extract(names_contents, reader)


def extract(dxls, reader):
    data_table = []
    signals = {}
    meta = None

    seen_files = set()
    expected_num_files = "unknown"

    skipped_files = []

    for name, dxl in dxls.items():
        file = reader(dxl)

        try:
            header, data = file.split("Begin data\n", 1)
        except Exception:
            print(
                f"ERROR: File {name} is not a DxL file. Missing 'Begin data'. Skipping..."
            )
            skipped_files.append((name, dxl))
            continue

        header = header.split("\n")
        files_match = FILE_OF_FILES_RE.search(header[13])
        num_file = "unknown"
        if files_match is None:
            print(
                f"WARNING: File {name} does not have a valid file count declaration. Expected 'x of y' in the header. This may lead to mixed up data. Continuing..."
            )
        else:
            num_file_match = int(files_match.group(1))

            if expected_num_files == "unknown":
                expected_num_files = int(files_match.group(2))
                print(
                    f"In file {name} found a header declaration of {expected_num_files} expected files"
                )
            else:
                if expected_num_files != int(files_match.group(2)):
                    print(
                        f"WARNING: File {name} has a different number of expected files than the previous files. This may lead to mixed up data. Continuing..."
                    )

            if num_file_match in seen_files:
                print(
                    f"WARNING: File {name} has a duplicate file number. This may lead to mixed up data. Continuing..."
                )
            else:
                num_file = num_file_match
                seen_files.add(num_file)

        print(f"Processing {name} - ({num_file} of {expected_num_files})")

        # meta_lines = header[0:11]

        # file_meta = {}
        # for line in meta_lines:
        #     key, value = line.strip().split(" : ")
        #     file_meta[key] = value

        # meta[num_file - 1] = file_meta
        try:
            data_blocks = data.split("\n\n")

            if len(data_blocks) < 2:
                print(
                    f"ERROR: In file {file} data blocks are missing or not splitted by two newlines. Skipping..."
                )
                skipped_files.append((name, dxl))
                continue

            data_lines, signal_blocks = data_blocks[0], data_blocks[1:]

            df_data = pd.read_csv(
                StringIO(data_lines), sep=",", index_col=0, header=None
            )
            df_data = df_data.transpose()
            df_data.columns = [column.strip().rstrip(":") for column in df_data.columns]
            data_table.append({"num_file": num_file, "data": df_data})

            signal_blocks = [
                remove_last_line_if_invalid(signal_block)
                for signal_block in signal_blocks
            ]

            num_blocks = len(signal_blocks)
            for num_block, block in enumerate(signal_blocks):
                try:
                    df = pd.read_csv(StringIO(block), sep=",", header=None)
                    block_name = df.iloc[0, 0].rstrip(":")
                except Exception:
                    print(
                        f"WARNING: Malformed data block {num_block + 1}/{num_blocks} in file {name}. Skipping..."
                    )
                    continue

                if block_name not in signals:
                    signals[block_name] = []

                df = df.drop(df.columns[0], axis=1)
                df = df.transpose()

                if block_name in DATA_BLOCKS_TO_READ_COORDS_FROM:
                    df[["x", "y"]] = df[0].apply(extract_coords).apply(pd.Series)

                    cols = df.columns.tolist()
                    cols = cols[-2:] + cols[:-2]
                    df = df[cols]

                    df = df.rename(columns={0: "label"})

                signals[block_name].append({"num_file": num_file, "data": df})

        except Exception as e:
            print(f"ERROR: File {name} is not a correct DxL file. Skipping...")
            skipped_files.append((name, dxl))
            continue

    if expected_num_files != "unknown":
        if len(seen_files) > expected_num_files:
            print(
                f"WARNING: Expected {expected_num_files} but found {len(seen_files)}. Some files may come from a different dataset"
            )
        if len(seen_files) < expected_num_files:
            missing_files = list(set(range(1, expected_num_files + 1)) - seen_files)
            missing_files = [str(x) for x in missing_files]
            print(
                f"WARNING: Expected {expected_num_files} but found {len(seen_files)}. Missing file numbers: {', '.join(missing_files)}"
            )

    data_table = [x for x in sorted(data_table, key=lambda x: x["num_file"])]
    num_unknown = len([x for x in data_table if x["num_file"] == "unknown"])

    if num_unknown < len(data_table):
        print(
            f"Files sorted in order: {', '.join([str(x['num_file']) for x in data_table])}"
        )

    if num_unknown > 0:
        print(
            f"WARNING: Data from {num_unknown} unnumbered files appended to the end of the data"
        )

    data_table = pd.concat([x["data"] for x in data_table]).reset_index(drop=True)

    if not all([col in data_table.columns for col in REQUIRED_DATA_TABLE_COLUMNS]):
        raise Exception(
            f"Missing required columns in data table. Required columns: {', '.join(REQUIRED_DATA_TABLE_COLUMNS)}"
        )

    if not all([col in signals for col in REQUIRED_DATA_BLOCKS]):
        raise Exception(
            f"Missing required data blocks. Required data blocks: {', '.join(REQUIRED_DATA_BLOCKS)}"
        )

    signals = {
        k: pd.concat(
            x["data"] for x in sorted(v, key=lambda x: x["num_file"])
        ).reset_index(drop=True)
        for k, v in signals.items()
    }

    print(f'Reading successful. Found data blocks: {", ".join(signals.keys())}')

    data_table = format_data_table_types(data_table)

    lessions = None
    if len(skipped_files) > 0:
        print(
            f"{len(skipped_files)} file couldn't be read as DxL. Trying as lessions..."
        )
        lessions = extract_lessions(skipped_files, reader)
    else:
        print("No lessions files found")

    return meta, data_table, signals, lessions


def format_data_table_types(data_table):
    data_table["rov LAT"] = data_table["rov LAT"].astype(float)
    data_table["ref LAT"] = data_table["ref LAT"].astype(float)
    data_table["peak2peak"] = data_table["peak2peak"].astype(float)
    data_table["Sample rate"] = data_table["Sample rate"].astype(float)

    return data_table


def remove_last_line_if_invalid(string):
    lines = string.split("\n")
    last_line = lines[-1]
    if last_line.startswith(","):
        return string
    else:
        return "\n".join(lines[:-1])
