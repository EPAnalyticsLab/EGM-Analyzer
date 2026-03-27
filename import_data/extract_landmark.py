import io
import os
import xml.etree.ElementTree as ET

import pandas as pd

from import_data.decode_raw_content import decode_raw_content

VERTICES_PATH_FROM_ROOT = "DIFBody/Volumes/Volume/Vertices"
FACES_PATH_FROM_ROOT = "DIFBody/Volumes/Volume/Polygons"


def extract_landmark(filenames, contents):
    xml_files = [f for f in filenames if f.endswith(".xml")]

    if len(xml_files) == 0:
        print(
            "WARNING: No .xml landmark file found. Geometry will not be plotted. Mapping will not be available"
        )
        return None, None

    xml_file = xml_files[0]

    if len(xml_files) > 1:
        print(
            f"WARNING: More than one .xml landmark file found. Using the first one found ({xml_file})"
        )
    else:
        print(f"Reading {xml_file} as landmark file")

    landmark_content = contents[filenames.index(xml_file)]
    landmark_string = decode_raw_content(landmark_content)

    return extract(landmark_string)


def extract_local_landmark(paths):
    xml_paths = [p for p in paths if p.endswith(".xml")]

    if len(xml_paths) == 0:
        print(
            "WARNING: No .xml landmark file found. Geometry will not be plotted. Mapping will not be available"
        )
        return None, None

    xml_path = xml_paths[0]
    filename = os.path.basename(xml_path)

    if len(xml_paths) > 1:
        print(
            f"WARNING: More than one .xml landmark file found. Using the first one found ({filename})"
        )
    else:
        print(f"Reading {filename} as landmark file")

    with open(xml_path, "r", encoding="utf-8") as f:
        landmark_string = f.read()
        return extract(landmark_string)


def extract(landmark_string):
    try:
        landmark_xml = ET.fromstring(landmark_string)

        vertices_text = landmark_xml.find(VERTICES_PATH_FROM_ROOT).text
        faces_text = landmark_xml.find(FACES_PATH_FROM_ROOT).text

        vertices = pd.read_csv(io.StringIO(vertices_text), sep=" ", header=None)
        faces = pd.read_csv(io.StringIO(faces_text), sep=" ", header=None)

        vertices = vertices.dropna(axis=1, how="all").dropna(axis=0, how="all")
        faces = faces.dropna(axis=1, how="all").dropna(axis=0, how="all")

        faces = faces.astype(int)
        vertices.columns = ["x", "y", "z"]
        faces.columns = ["v1", "v2", "v3"]
        faces = faces - 1

    except Exception as e:
        print(
            f"ERROR: Could not read landmark file. Check if the file has the correct format. Error: {e}"
        )
        return None, None

    print(
        f"Landmark read successfully. Found {len(vertices)} vertices and {len(faces)} faces"
    )
    return vertices, faces
