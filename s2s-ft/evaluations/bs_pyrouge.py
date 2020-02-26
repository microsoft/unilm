from __future__ import print_function, unicode_literals, division

import os
import re
import codecs
import platform

from subprocess import check_output
from tempfile import mkdtemp
from functools import partial

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

from pyrouge.utils import log
from pyrouge.utils.file_utils import verify_dir


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


class DirectoryProcessor:

    @staticmethod
    def process(input_dir, output_dir, function):
        """
        Apply function to all files in input_dir and save the resulting ouput
        files in output_dir.

        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger = log.get_global_console_logger()
        logger.info("Processing files in {}.".format(input_dir))
        input_file_names = os.listdir(input_dir)
        for input_file_name in input_file_names:
            input_file = os.path.join(input_dir, input_file_name)
            with codecs.open(input_file, "r", encoding="UTF-8") as f:
                input_string = f.read()
            output_string = function(input_string)
            output_file = os.path.join(output_dir, input_file_name)
            with codecs.open(output_file, "w", encoding="UTF-8") as f:
                f.write(clean(output_string.lower()))
        logger.info("Saved processed files to {}.".format(output_dir))


class Rouge155(object):
    """
    This is a wrapper for the ROUGE 1.5.5 summary evaluation package.
    This class is designed to simplify the evaluation process by:

        1) Converting summaries into a format ROUGE understands.
        2) Generating the ROUGE configuration file automatically based
            on filename patterns.

    This class can be used within Python like this:

    rouge = Rouge155()
    rouge.system_dir = 'test/systems'
    rouge.model_dir = 'test/models'

    # The system filename pattern should contain one group that
    # matches the document ID.
    rouge.system_filename_pattern = 'SL.P.10.R.11.SL062003-(\d+).html'

    # The model filename pattern has '#ID#' as a placeholder for the
    # document ID. If there are multiple model summaries, pyrouge
    # will use the provided regex to automatically match them with
    # the corresponding system summary. Here, [A-Z] matches
    # multiple model summaries for a given #ID#.
    rouge.model_filename_pattern = 'SL.P.10.R.[A-Z].SL062003-#ID#.html'

    rouge_output = rouge.evaluate()
    print(rouge_output)
    output_dict = rouge.output_to_dict(rouge_ouput)
    print(output_dict)
    ->    {'rouge_1_f_score': 0.95652,
         'rouge_1_f_score_cb': 0.95652,
         'rouge_1_f_score_ce': 0.95652,
         'rouge_1_precision': 0.95652,
        [...]


    To evaluate multiple systems:

        rouge = Rouge155()
        rouge.system_dir = '/PATH/TO/systems'
        rouge.model_dir = 'PATH/TO/models'
        for system_id in ['id1', 'id2', 'id3']:
            rouge.system_filename_pattern = \
                'SL.P/.10.R.{}.SL062003-(\d+).html'.format(system_id)
            rouge.model_filename_pattern = \
                'SL.P.10.R.[A-Z].SL062003-#ID#.html'
            rouge_output = rouge.evaluate(system_id)
            print(rouge_output)

    """

    def __init__(self, rouge_dir=None, rouge_args=None, temp_dir=None):
        """
        Create a Rouge155 object.

            rouge_dir:  Directory containing Rouge-1.5.5.pl
            rouge_args: Arguments to pass through to ROUGE if you
                        don't want to use the default pyrouge
                        arguments.

        """
        self.temp_dir = temp_dir
        self.log = log.get_global_console_logger()
        self.__set_dir_properties()
        self._config_file = None
        self._settings_file = self.__get_config_path()
        self.__set_rouge_dir(rouge_dir)
        self.args = self.__clean_rouge_args(rouge_args)
        self._system_filename_pattern = None
        self._model_filename_pattern = None

    def save_home_dir(self):
        config = ConfigParser()
        section = 'pyrouge settings'
        config.add_section(section)
        config.set(section, 'home_dir', self._home_dir)
        with open(self._settings_file, 'w') as f:
            config.write(f)
        self.log.info("Set ROUGE home directory to {}.".format(self._home_dir))

    @property
    def settings_file(self):
        """
        Path of the setttings file, which stores the ROUGE home dir.

        """
        return self._settings_file

    @property
    def bin_path(self):
        """
        The full path of the ROUGE binary (although it's technically
        a script), i.e. rouge_home_dir/ROUGE-1.5.5.pl

        """
        if self._bin_path is None:
            raise Exception(
                "ROUGE path not set. Please set the ROUGE home directory "
                "and ensure that ROUGE-1.5.5.pl exists in it.")
        return self._bin_path

    @property
    def system_filename_pattern(self):
        """
        The regular expression pattern for matching system summary
        filenames. The regex string.

        E.g. "SL.P.10.R.11.SL062003-(\d+).html" will match the system
        filenames in the SPL2003/system folder of the ROUGE SPL example
        in the "sample-test" folder.

        Currently, there is no support for multiple systems.

        """
        return self._system_filename_pattern

    @system_filename_pattern.setter
    def system_filename_pattern(self, pattern):
        self._system_filename_pattern = pattern

    @property
    def model_filename_pattern(self):
        """
        The regular expression pattern for matching model summary
        filenames. The pattern needs to contain the string "#ID#",
        which is a placeholder for the document ID.

        E.g. "SL.P.10.R.[A-Z].SL062003-#ID#.html" will match the model
        filenames in the SPL2003/system folder of the ROUGE SPL
        example in the "sample-test" folder.

        "#ID#" is a placeholder for the document ID which has been
        matched by the "(\d+)" part of the system filename pattern.
        The different model summaries for a given document ID are
        matched by the "[A-Z]" part.

        """
        return self._model_filename_pattern

    @model_filename_pattern.setter
    def model_filename_pattern(self, pattern):
        self._model_filename_pattern = pattern

    @property
    def config_file(self):
        return self._config_file

    @config_file.setter
    def config_file(self, path):
        config_dir, _ = os.path.split(path)
        verify_dir(config_dir, "configuration file")
        self._config_file = path

    def split_sentences(self):
        """
        ROUGE requires texts split into sentences. In case the texts
        are not already split, this method can be used.

        """
        from pyrouge.utils.sentence_splitter import PunktSentenceSplitter
        self.log.info("Splitting sentences.")
        ss = PunktSentenceSplitter()

        def sent_split_to_string(s): return "\n".join(ss.split(s))
        process_func = partial(
            DirectoryProcessor.process, function=sent_split_to_string)
        self.__process_summaries(process_func)

    @staticmethod
    def convert_summaries_to_rouge_format(input_dir, output_dir):
        """
        Convert all files in input_dir into a format ROUGE understands
        and saves the files to output_dir. The input files are assumed
        to be plain text with one sentence per line.

            input_dir:  Path of directory containing the input files.
            output_dir: Path of directory in which the converted files
                        will be saved.

        """
        DirectoryProcessor.process(
            input_dir, output_dir, Rouge155.convert_text_to_rouge_format)

    @staticmethod
    def convert_text_to_rouge_format(text, title="dummy title"):
        """
        Convert a text to a format ROUGE understands. The text is
        assumed to contain one sentence per line.

            text:   The text to convert, containg one sentence per line.
            title:  Optional title for the text. The title will appear
                    in the converted file, but doesn't seem to have
                    any other relevance.

        Returns: The converted text as string.

        """
        sentences = text.split("\n")
        sent_elems = [
            "<a name=\"{i}\">[{i}]</a> <a href=\"#{i}\" id={i}>"
            "{text}</a>".format(i=i, text=sent)
            for i, sent in enumerate(sentences, start=1)]
        html = """<html>
<head>
<title>{title}</title>
</head>
<body bgcolor="white">
{elems}
</body>
</html>""".format(title=title, elems="\n".join(sent_elems))

        return html

    @staticmethod
    def write_config_static(system_dir, system_filename_pattern,
                            model_dir, model_filename_pattern,
                            config_file_path, system_id=None):
        """
        Write the ROUGE configuration file, which is basically a list
        of system summary files and their corresponding model summary
        files.

        pyrouge uses regular expressions to automatically find the
        matching model summary files for a given system summary file
        (cf. docstrings for system_filename_pattern and
        model_filename_pattern).

            system_dir:                 Path of directory containing
                                        system summaries.
            system_filename_pattern:    Regex string for matching
                                        system summary filenames.
            model_dir:                  Path of directory containing
                                        model summaries.
            model_filename_pattern:     Regex string for matching model
                                        summary filenames.
            config_file_path:           Path of the configuration file.
            system_id:                  Optional system ID string which
                                        will appear in the ROUGE output.

        """
        system_filenames = [f for f in os.listdir(system_dir)]
        system_models_tuples = []

        system_filename_pattern = re.compile(system_filename_pattern)
        for system_filename in sorted(system_filenames):
            match = system_filename_pattern.match(system_filename)
            if match:
                id = match.groups(0)[0]
                model_filenames = [model_filename_pattern.replace('#ID#', id)]
                # model_filenames = Rouge155.__get_model_filenames_for_id(
                #     id, model_dir, model_filename_pattern)
                system_models_tuples.append(
                    (system_filename, sorted(model_filenames)))
        if not system_models_tuples:
            raise Exception(
                "Did not find any files matching the pattern {} "
                "in the system summaries directory {}.".format(
                    system_filename_pattern.pattern, system_dir))

        with codecs.open(config_file_path, 'w', encoding='utf-8') as f:
            f.write('<ROUGE-EVAL version="1.55">')
            for task_id, (system_filename, model_filenames) in enumerate(
                    system_models_tuples, start=1):

                eval_string = Rouge155.__get_eval_string(
                    task_id, system_id,
                    system_dir, system_filename,
                    model_dir, model_filenames)
                f.write(eval_string)
            f.write("</ROUGE-EVAL>")

    def write_config(self, config_file_path=None, system_id=None):
        """
        Write the ROUGE configuration file, which is basically a list
        of system summary files and their matching model summary files.

        This is a non-static version of write_config_file_static().

            config_file_path:   Path of the configuration file.
            system_id:          Optional system ID string which will
                                appear in the ROUGE output.

        """
        if not system_id:
            system_id = 1
        if (not config_file_path) or (not self._config_dir):
            self._config_dir = mkdtemp(dir=self.temp_dir)
            config_filename = "rouge_conf.xml"
        else:
            config_dir, config_filename = os.path.split(config_file_path)
            verify_dir(config_dir, "configuration file")
        self._config_file = os.path.join(self._config_dir, config_filename)
        Rouge155.write_config_static(
            self._system_dir, self._system_filename_pattern,
            self._model_dir, self._model_filename_pattern,
            self._config_file, system_id)
        self.log.info(
            "Written ROUGE configuration to {}".format(self._config_file))

    def evaluate(self, system_id=1, rouge_args=None):
        """
        Run ROUGE to evaluate the system summaries in system_dir against
        the model summaries in model_dir. The summaries are assumed to
        be in the one-sentence-per-line HTML format ROUGE understands.

            system_id:  Optional system ID which will be printed in
                        ROUGE's output.

        Returns: Rouge output as string.

        """
        self.write_config(system_id=system_id)
        options = self.__get_options(rouge_args)
        command = [self._bin_path] + options
        self.log.info(
            "Running ROUGE with command {}".format(" ".join(command)))
        rouge_output = check_output(command).decode("UTF-8")
        return rouge_output

    def convert_and_evaluate(self, system_id=1,
                             split_sentences=False, rouge_args=None):
        """
        Convert plain text summaries to ROUGE format and run ROUGE to
        evaluate the system summaries in system_dir against the model
        summaries in model_dir. Optionally split texts into sentences
        in case they aren't already.

        This is just a convenience method combining
        convert_summaries_to_rouge_format() and evaluate().

            split_sentences:    Optional argument specifying if
                                sentences should be split.
            system_id:          Optional system ID which will be printed
                                in ROUGE's output.

        Returns: ROUGE output as string.

        """
        if split_sentences:
            self.split_sentences()
        self.__write_summaries()
        rouge_output = self.evaluate(system_id, rouge_args)
        return rouge_output

    def output_to_dict(self, output):
        """
        Convert the ROUGE output into python dictionary for further
        processing.

        """
        # 0 ROUGE-1 Average_R: 0.02632 (95%-conf.int. 0.02632 - 0.02632)
        pattern = re.compile(
            r"(\d+) (ROUGE-\S+) (Average_\w): (\d.\d+) "
            r"\(95%-conf.int. (\d.\d+) - (\d.\d+)\)")
        results = {}
        for line in output.split("\n"):
            match = pattern.match(line)
            if match:
                sys_id, rouge_type, measure, result, conf_begin, conf_end = \
                    match.groups()
                measure = {
                    'Average_R': 'recall',
                    'Average_P': 'precision',
                    'Average_F': 'f_score'
                }[measure]
                rouge_type = rouge_type.lower().replace("-", '_')
                key = "{}_{}".format(rouge_type, measure)
                results[key] = float(result)
                results["{}_cb".format(key)] = float(conf_begin)
                results["{}_ce".format(key)] = float(conf_end)
        return results

    ###################################################################
    # Private methods

    def __set_rouge_dir(self, home_dir=None):
        """
        Verfify presence of ROUGE-1.5.5.pl and data folder, and set
        those paths.

        """
        if not home_dir:
            self._home_dir = self.__get_rouge_home_dir_from_settings()
        else:
            self._home_dir = home_dir
            self.save_home_dir()
        self._bin_path = os.path.join(self._home_dir, 'ROUGE-1.5.5.pl')
        self.data_dir = os.path.join(self._home_dir, 'data')
        if not os.path.exists(self._bin_path):
            raise Exception(
                "ROUGE binary not found at {}. Please set the "
                "correct path by running pyrouge_set_rouge_path "
                "/path/to/rouge/home.".format(self._bin_path))

    def __get_rouge_home_dir_from_settings(self):
        config = ConfigParser()
        with open(self._settings_file) as f:
            if hasattr(config, "read_file"):
                config.read_file(f)
            else:
                # use deprecated python 2.x method
                config.readfp(f)
        rouge_home_dir = config.get('pyrouge settings', 'home_dir')
        return rouge_home_dir

    @staticmethod
    def __get_eval_string(
            task_id, system_id,
            system_dir, system_filename,
            model_dir, model_filenames):
        """
        ROUGE can evaluate several system summaries for a given text
        against several model summaries, i.e. there is an m-to-n
        relation between system and model summaries. The system
        summaries are listed in the <PEERS> tag and the model summaries
        in the <MODELS> tag. pyrouge currently only supports one system
        summary per text, i.e. it assumes a 1-to-n relation between
        system and model summaries.

        """
        peer_elems = "<P ID=\"{id}\">{name}</P>".format(
            id=system_id, name=system_filename)

        model_elems = ["<M ID=\"{id}\">{name}</M>".format(
            id=chr(65 + i), name=name)
            for i, name in enumerate(model_filenames)]

        model_elems = "\n\t\t\t".join(model_elems)
        eval_string = """
    <EVAL ID="{task_id}">
        <MODEL-ROOT>{model_root}</MODEL-ROOT>
        <PEER-ROOT>{peer_root}</PEER-ROOT>
        <INPUT-FORMAT TYPE="SEE">
        </INPUT-FORMAT>
        <PEERS>
            {peer_elems}
        </PEERS>
        <MODELS>
            {model_elems}
        </MODELS>
    </EVAL>
""".format(
            task_id=task_id,
            model_root=model_dir, model_elems=model_elems,
            peer_root=system_dir, peer_elems=peer_elems)
        return eval_string

    def __process_summaries(self, process_func):
        """
        Helper method that applies process_func to the files in the
        system and model folders and saves the resulting files to new
        system and model folders.

        """
        temp_dir = mkdtemp(dir=self.temp_dir)
        new_system_dir = os.path.join(temp_dir, "system")
        os.mkdir(new_system_dir)
        new_model_dir = os.path.join(temp_dir, "model")
        os.mkdir(new_model_dir)
        self.log.info(
            "Processing summaries. Saving system files to {} and "
            "model files to {}.".format(new_system_dir, new_model_dir))
        process_func(self._system_dir, new_system_dir)
        process_func(self._model_dir, new_model_dir)
        self._system_dir = new_system_dir
        self._model_dir = new_model_dir

    def __write_summaries(self):
        self.log.info("Writing summaries.")
        self.__process_summaries(self.convert_summaries_to_rouge_format)

    @staticmethod
    def __get_model_filenames_for_id(id, model_dir, model_filenames_pattern):
        pattern = re.compile(model_filenames_pattern.replace('#ID#', id))
        model_filenames = [
            f for f in os.listdir(model_dir) if pattern.match(f)]
        if not model_filenames:
            raise Exception(
                "Could not find any model summaries for the system"
                " summary with ID {}. Specified model filename pattern was: "
                "{}".format(id, model_filenames_pattern))
        return model_filenames

    def __get_options(self, rouge_args=None):
        """
        Get supplied command line arguments for ROUGE or use default
        ones.

        """
        if self.args:
            options = self.args.split()
        elif rouge_args:
            options = rouge_args.split()
        else:
            options = [
                '-e', self._data_dir,
                '-c', 95,
                # '-2',
                # '-1',
                # '-U',
                '-m',
                # '-v',
                '-r', 1000,
                '-n', 2,
                # '-w', 1.2,
                '-a',
            ]
            options = list(map(str, options))

        options = self.__add_config_option(options)
        return options

    def __create_dir_property(self, dir_name, docstring):
        """
        Generate getter and setter for a directory property.

        """
        property_name = "{}_dir".format(dir_name)
        private_name = "_" + property_name
        setattr(self, private_name, None)

        def fget(self):
            return getattr(self, private_name)

        def fset(self, path):
            verify_dir(path, dir_name)
            setattr(self, private_name, path)

        p = property(fget=fget, fset=fset, doc=docstring)
        setattr(self.__class__, property_name, p)

    def __set_dir_properties(self):
        """
        Automatically generate the properties for directories.

        """
        directories = [
            ("home", "The ROUGE home directory."),
            ("data", "The path of the ROUGE 'data' directory."),
            ("system", "Path of the directory containing system summaries."),
            ("model", "Path of the directory containing model summaries."),
        ]
        for (dirname, docstring) in directories:
            self.__create_dir_property(dirname, docstring)

    def __clean_rouge_args(self, rouge_args):
        """
        Remove enclosing quotation marks, if any.

        """
        if not rouge_args:
            return
        quot_mark_pattern = re.compile('"(.+)"')
        match = quot_mark_pattern.match(rouge_args)
        if match:
            cleaned_args = match.group(1)
            return cleaned_args
        else:
            return rouge_args

    def __add_config_option(self, options):
        return options + [self._config_file]

    def __get_config_path(self):
        if platform.system() == "Windows":
            parent_dir = os.getenv("APPDATA")
            config_dir_name = "pyrouge"
        elif os.name == "posix":
            parent_dir = os.path.expanduser("~")
            config_dir_name = ".pyrouge"
        else:
            parent_dir = os.path.dirname(__file__)
            config_dir_name = ""
        config_dir = os.path.join(parent_dir, config_dir_name)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        return os.path.join(config_dir, 'settings.ini')


if __name__ == "__main__":
    import argparse
    from utils.argparsers import rouge_path_parser

    parser = argparse.ArgumentParser(parents=[rouge_path_parser])
    args = parser.parse_args()

    rouge = Rouge155(args.rouge_home)
    rouge.save_home_dir()
