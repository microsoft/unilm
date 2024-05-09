from .harness_task import HarnessBaseTask


SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def create_mmlu_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """
    return {f"hendrycksTest-{sub}": create_task(f"hendrycksTest-{sub}") for sub in SUBJECTS}


def create_task(subject):
    class HendrycksTest(GeneralHendrycksTest):
        def set_dataname(self):
            self.dataname = f"{subject}"

    return HendrycksTest

class GeneralHendrycksTest(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 4

    def preprocess_example(self, example):
        # find the last occurence of "Queston:" in example["text"], and remove everything before it
        # this is to remove the context
        # last_question = example["text"].rfind("Question:")
        # example["text"] = example["text"][last_question:]
        input_str = [example["text"]] * self.class_num
        answer_str = [' ' + item for item in example["choices"]]
        label = example["gold"]
        return input_str, answer_str, label
