import copy


def convert_dict(target_dict, pre_key=""):
    converted_dict = {}
    tmp_converted_dict = {}
    for key, value in target_dict.items():
        if isinstance(value, dict):
            tmp_converted_dict.update(convert_dict(value, key))
        else:
            if pre_key:
                converted_dict[f"{pre_key}/{key}"] = value
            else:
                converted_dict[key] = value
    for key, value in tmp_converted_dict.items():
        if pre_key:
            converted_dict[f"{pre_key}/{key}"] = value
        else:
            converted_dict[key] = value

    return converted_dict


def convert_dicts(target_dicts):
    converted_dict_list = []
    for target_dict in target_dicts:
        converted_dict_list.append(convert_dict(target_dict))
    return converted_dict_list


def expand_and_overwrite_dict(original_dict, new_dict):
    expanded_dict = copy.deepcopy(original_dict)
    for key, value in new_dict.items():
        expanded_dict[key] = value
    return expanded_dict
