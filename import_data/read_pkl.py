import pickle
import tkinter as tk
from tkinter import filedialog


def read_pkl():
    root = tk.Tk()
    root.withdraw()

    paths = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])

    root.destroy()

    if paths is None or len(paths) == 0:
        raise Exception("No files selected")

    print("Importing from", paths)
    with open(paths, "rb") as f:
        data = pickle.load(f)
        print("Import complete")

    return data
