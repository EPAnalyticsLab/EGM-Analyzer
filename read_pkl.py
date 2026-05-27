import pandas as pd
from io import StringIO

REQUIRED_LESSIONS_COLUMNS = ["x", "y", "z"]


def extract_lessions(files, reader):
    lessions = []

    for name, file in files:
        print(f"Extracting lessions from {name}")
        file = reader(file)
        blocks = file.split("\n\n")

        for i, block in enumerate(blocks):
            lines = block.split("\n")
            lines = [line for line in lines if line.count(",") > 1]

            try:
                df = pd.read_csv(StringIO("\n".join(lines)), sep=",")
            except Exception as e:
                print(
                    f"WARNING: Malformed lession block ({i} of {len(blocks)}) in file {name}. Skipping the block..."
                )
                continue

            if not all(col in df.columns for col in REQUIRED_LESSIONS_COLUMNS):
                print(
                    f"WARNING: Malformed lession block ({i} of {len(blocks)}) in file {name}. Skipping the block..."
                )
                continue

            lessions.append(df)

    if len(lessions) == 0:
        print("WARNING: No lessions found")
        return None

    lessions = pd.concat(lessions)
    print(f"Found {len(lessions)} lessions")

    return lessions
