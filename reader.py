import uproot3 as uproot
import numpy as np
import pandas as pd


def _show_uproot_tree(obj, max_key_len=12, sep='/', indent=0) -> None:
    width = max_key_len + len(sep)
    start_line = False
    if isinstance(obj, uproot.rootio.ROOTDirectory):
        print('TFile: ' + obj.name.decode('utf-8'))
        start_line = True
        indent = 2
    elif issubclass(type(obj), uproot.tree.TTreeMethods):
        print('TTree: ' + obj.name.decode('utf-8'))
        start_line = True
        indent = 4
    else:
        if len(obj.keys()) > 0:
            indent += width
            s = obj.name.decode('utf-8')[:max_key_len]
            print(s + ' ' * (max_key_len - len(s)) + sep, end='')
        else:
            print(obj.name.decode('utf-8'))

    if len(obj.keys()) > 0:
        for i, key in enumerate(obj.keys()):
            if i > 0 or start_line:
                print(' ' * indent, end='')
            _show_uproot_tree(obj[key], indent=indent)
        indent -= width


def show_uproot_tree(filename: str, max_key_len=12, sep='/', indent=0) -> None:
    with uproot.open(filename) as f:
        _show_uproot_tree(f, max_key_len, sep, indent)
