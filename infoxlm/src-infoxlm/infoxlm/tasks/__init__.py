import argparse
import importlib
import os

from fairseq.tasks import TASK_REGISTRY



# automatically import any Python files in the tasks/ directory
for file in os.listdir(os.path.dirname(__file__)):
  if file.endswith('.py') and not file.startswith('_'):
    task_name = file[:file.find('.py')]
    importlib.import_module('infoxlm.tasks.' + task_name)

    # expose `task_parser` for sphinx
    if task_name in TASK_REGISTRY:
      parser = argparse.ArgumentParser(add_help=False)
      group_task = parser.add_argument_group('Task name')
      # fmt: off
      group_task.add_argument('--task', metavar=task_name,
                              help='Enable this task with: ``--task=' + task_name + '``')
      # fmt: on
      group_args = parser.add_argument_group('Additional command-line arguments')
      TASK_REGISTRY[task_name].add_args(group_args)
      globals()[task_name + '_parser'] = parser