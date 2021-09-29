"""I/O"""

def _lines_gen_from_single_file(filename):
  with open(filename) as fp:
    for line in fp: yield line.strip()


def lines_gen(*filenames):
  for ret in zip(*map(_lines_gen_from_single_file, filenames)): yield ret