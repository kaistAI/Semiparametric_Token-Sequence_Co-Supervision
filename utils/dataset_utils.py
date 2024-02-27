import json, pickle, random
import pandas as pd

def sample_dict(dictionary, number):
    sampled = {}
    for key in dictionary.keys():
        sampled[key] = dictionary[key][:number]
    return sampled

def add_instruction_rettoken(
    data_path, data_type, do_swap, dataset_config
):
    data_list = []
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".json"):
        df = json.load(open(data_path))
    elif data_path.endswith(".pickle"):
        df = pickle.load(open(data_path, "rb"))
    else:
        assert False, f"Check data_path: {data_path}"
    if dataset_config.train_sample > 0:
        df = sample_dict(df, dataset_config.train_sample)

    if "evidence" in df.keys():
        if "hard_negs" in df.keys():
            for _input, _output, _ev, _hardneg, _input_ids in zip(
                df["input"],
                df["output"],
                df["evidence"],
                df["hard_negs"],
                df["input_ids"],
            ):
                target_idxs = [i for i, val in enumerate(_input_ids) if val >= 32003]
                if len(target_idxs) > 0 or dataset_config.allow_noret:
                    for item in _ev:
                        _output = _output.replace(f"[k_{item}]", "[Ret]")
                    data_list.append(
                        {
                            "instruction": _input,
                            "input": "",
                            "output": _output,
                            "q": _input,
                            "evidence": [f"[k_{item}]" for item in _ev],
                            "target_idxs": target_idxs,
                            "hard_negs": _hardneg,
                        }
                    )
        else:
            for _input, _output, _ev, _input_ids in zip(
                df["input"], df["output"], df["evidence"], df["input_ids"]
            ):
                target_idxs = [i for i, val in enumerate(_input_ids) if val >= 32003]
                if len(target_idxs) > 0 or dataset_config.allow_noret:
                    for item in _ev:
                        _output = _output.replace(f"[k_{item}]", "[Ret]")
                    data_list.append(
                        {
                            "instruction": _input,
                            "input": "",
                            "output": _output,
                            "q": _input,
                            "evidence": [f"[k_{item}]" for item in _ev],
                            "target_idxs": target_idxs,
                        }
                )
    else:
        for _input, _output in zip(df["input"], df["output"]):
            data_list.append(
                {
                    "instruction": _input,
                    "input": "",
                    "output": _output,
                    "q": _input,
                }
            )
    return data_list
