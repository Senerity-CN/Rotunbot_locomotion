from abc import ABC
import os
from engineai_gym import ENGINEAI_GYM_ROOT_DIR
from engineai_gym.tester.loggers.logger_base import LoggerBase
from engineai_rl_lib.dict_operations import expand_and_overwrite_dict
from engineai_rl_lib.files_and_dirs import import_modules_of_specific_type_from_path


class TesterTypeBase(ABC):
    def __init__(
        self,
        name,
        tester_and_parent_class_files,
        loggers,
        env,
        time,
        test_dir,
        extra_args,
    ):
        self.imported_classes = {}
        for file in tester_and_parent_class_files:
            current_file_directory = os.path.join(os.path.dirname(file), "loggers")
            if os.path.exists(current_file_directory):
                self.imported_classes = expand_and_overwrite_dict(
                    self.imported_classes,
                    import_modules_of_specific_type_from_path(
                        ENGINEAI_GYM_ROOT_DIR, current_file_directory, LoggerBase
                    ),
                )
        self.name = name
        self.env = env
        if self.__class__.__name__ != "TesterBase":
            self.test_dir = os.path.join(test_dir, name)
        self.loggers = []
        for key, value in loggers.items():
            logger_class = self.imported_classes[value]
            self.loggers.append(
                logger_class(key, env, time, os.path.join(test_dir, name), extra_args)
            )

    def set_goals(self):
        return self.env.goal_dict

    def start_record_video(self):
        self.env.start_recording_video()

    def end_and_save_recording_video(self):
        self.env.end_and_save_recording_video(self.name + ".mp4")
