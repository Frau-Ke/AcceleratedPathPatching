import os
import pandas as pd
import json
import argparse

_PATH = None

def set_PATH(value):
    global _PATH
    _PATH = value

def get_PATH():
    return _PATH

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def store_df(df, out_path, name):
    create_folder(out_path)

    file_path =  os.path.join(out_path, name)
    if file_path.endswith(".csv"):
        df.to_csv(file_path, index=False)
    elif file_path.endswith(".xlsx"):
        df.to_excel(file_path, index=False)
    elif file_path.endswith(".json"):
        df.to_json(file_path, orient="records", indent=4)
    else:
        raise ValueError("Unsupported file format. Use .csv, .xlsx, or .json")

    print(f"DataFrame saved as {name} at {out_path}")


def read_df(in_path, name):
    file_path =  os.path.join(in_path, name)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path, orient="records", indent=4)
    else:
        raise ValueError("Unsupported file format. Use .csv, .xlsx, or .json")

    print(f"DataFrame saved as {name} at {in_path}")
    return df

def save_panda_to_text(df, out_path, name):
    create_folder(out_path)
    file_path = os.path.join(out_path, name)
    if file_path.exists():
        file_path.unlink()
    with open(file_path, 'a') as f:
        f.write(df.to_string(header=False, index=False)) 


def save_dict(dict, out_path, name):
    create_folder(out_path)
    file_path =  os.path.join(out_path, name)
    with open(file_path, 'w') as fp:
        json.dump(dict, fp)


def save_circuit(circuit, out_path, name): 
    # Save to file
    create_folder(out_path)
    circuit = dict(sorted(circuit.items()))
    with open(os.path.join(out_path, name), 'w') as file:
        file.write(str(circuit))


def load_circuit(out_path, name):
    with open(os.path.join(out_path, name), "r") as f:
        circuit = eval(f.read())
    return circuit


def save_img(fig, out_path, name):
    create_folder(out_path)
    file_path =  os.path.join(out_path, name)
    fig.savefig(file_path, bbox_inches='tight')
    print("save img at", file_path)


def save_parser_information(args, subfolder, name):
    with open(os.path.join(subfolder, name), 'w') as f:
        json.dump(vars(args), f, indent=2)

def load_parser_information(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return argparse.Namespace(**data)