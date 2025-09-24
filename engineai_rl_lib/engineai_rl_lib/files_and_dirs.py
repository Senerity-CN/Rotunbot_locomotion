import os
import importlib
import importlib.util


def get_module_path_from_folders_in_dir(root_path, dir, prefix="", exception=""):
    item_paths = get_py_file_paths_from_dir(dir)
    import_modules = {}
    # List all items in the directory
    for item_path in item_paths:
        # Check if the item is a directory and contains an __init__.py file
        if (
            os.path.isdir(item_path)
            and "__init__.py" in os.listdir(item_path)
            and item_path.rsplit("/", 1)[-1].startswith(prefix)
        ):
            module_path = get_module_path(root_path, item_path)
            module_name = module_path.rsplit(".", 1)[-1]
            if module_name != exception:
                # Import the package using importlib
                import_modules[module_name] = module_path
    return import_modules


def get_module_path_from_files_in_dir(root_path, dir, prefix="", exception=""):
    item_paths = get_py_file_paths_from_dir(dir)
    import_modules = {}
    # List all items in the directory
    for item_path in item_paths:
        # Check if the item is a directory and contains an __init__.py file
        module_path, module_name = get_module_path_from_file(
            root_path, item_path, prefix, exception
        )
        if module_path is not None:
            # Import the package using importlib
            import_modules[module_name] = module_path
    return import_modules


def get_module_path_from_file(root_path, file_path, prefix="", exception=""):
    if file_path.endswith(".py") and file_path.rsplit("/", 1)[-1].startswith(prefix):
        module_path = get_module_path(root_path, file_path)
        module_name = module_path.rsplit(".", 1)[-1]
        if module_name != exception:
            return module_path, module_name
        else:
            return None, None
    else:
        return None, None


def get_py_file_paths_from_dir(dir, exception=[]):
    # Get the absolute path of the files
    directory_path = os.path.abspath(dir)
    item_paths = []
    for item in os.listdir(directory_path):
        if item not in exception and item.endswith(".py"):
            item_paths.append(os.path.join(directory_path, item))
    return item_paths


def get_folder_paths_from_dir(dir):
    # Get the absolute path of the directory
    directory_path = os.path.abspath(dir)
    folder_paths = []
    for item in os.listdir(directory_path):
        path = os.path.join(directory_path, item)
        if os.path.isdir(path):
            folder_paths.append(path)
    return folder_paths


def get_module_path(root_path, item_path):
    if os.path.isfile(item_path) and item_path.endswith(".py"):
        item_path_without_suffix = item_path[:-3]
    else:
        item_path_without_suffix = item_path
    module = item_path_without_suffix.replace(root_path, "")[1:]
    return module.replace("/", ".")


def import_modules_of_specific_type_from_path(package_root, path, end_class=object):
    import_modules = get_module_path_from_files_in_dir(package_root, path)
    imported_classes = {}
    for module_name, module_path in import_modules.items():
        module = importlib.import_module(module_path)
        for attribute in dir(module):
            attr_value = getattr(module, attribute)
            if isinstance(attr_value, type):
                if issubclass(attr_value, end_class) and attr_value != end_class:
                    imported_classes[module_name] = getattr(module, attribute)
    return imported_classes


def import_attr_from_file_path(root_path, file_path, attr_name):
    module_full_name, module_name = get_module_path_from_file(root_path, file_path)
    spec = importlib.util.spec_from_file_location(
        location=file_path,
        name=module_full_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)
