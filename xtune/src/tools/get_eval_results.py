import argparse
import json
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--log_path",
        default=None,
        type=str,
        required=True,
        help="path to evaluation log file.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="task name",
    )
    args = parser.parse_args()

    all_res_terms = []

    all_best = 0.0
    all_final_res = None
    for path in sorted(glob.glob(args.log_path), key=lambda x: (len(x), x)):
        if args.task_name != "squad" and path.find("squad") != -1:
            continue
        # print(path)
        best = 0.0
        final_res = None
        lines = open(path).readlines()
        for line in lines:
            line = line.strip()
            if args.task_name == "squad":
                json_str = line.split('\t')[0]
            else:
                json_str = line.split('\t')[1]
            # print(json_str)
            res = json.loads(json_str)
            if args.task_name.lower() == "xnli" or args.task_name.lower() == "pawsx":
                if res["valid_avg"]["acc"] > best:
                    # print(res)
                    best = res["valid_avg"]["acc"]
                    final_res = res
            elif args.task_name.lower() in ["panx", "udpos"]:
                if "dev_avg" not in res or res["dev_avg"]["f1"] > best:
                    # print(res)
                    if "dev_avg" in res:
                        best = res["dev_avg"]["f1"]
                    final_res = res
            elif args.task_name.lower() in ["xquad", "mlqa", "tydiqa", "squad"]:
                final_res = res
        if args.task_name in ["xnli", "pawsx", "udpos", "panx"]:
            if best > all_best:
                all_best = best
                all_final_res = final_res
        elif args.task_name == "tydiqa":
            if final_res["dev_en"]["f1"] > all_best:
                all_best = final_res["dev_en"]["f1"]
                all_final_res = final_res
        elif args.task_name == "mlqa":
            if final_res["dev_avg"]["f1"] > all_best:
                all_best = final_res["dev_avg"]["f1"]
                all_final_res = final_res
        else:
            pass

        res_terms = []
        try:
            if args.task_name == "xnli":
                # order = ["en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]
                order = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
                acc = {}
                final_output = ""
                for k in final_res:
                    if k.find("-") == -1:
                        continue
                    mode, lang = k.split("-")
                    if mode == "test":
                        # final_output += lang + " " + str(final_res[k]["acc"]) + '\t'
                        acc[lang] = round(final_res[k]["acc"] * 100, 2)
                gap_sum = 0.0
                for lang in order:
                    res_terms.append(acc[lang])
                    gap_sum += acc["en"] - acc[lang]
                res_terms.append(round(final_res["test_avg"]["acc"] * 100, 2))
                res_terms.append(round(final_res["valid_avg"]["acc"] * 100, 2))

                res_terms.append(round(gap_sum / (len(res_terms) - 3), 2))

                for term in res_terms:
                    final_output += str(term) + ","
                print(final_output[:-1])
            elif args.task_name == "pawsx":
                # order = ["de", "en", "es", "fr", "ja", "ko", "zh"]
                order = ["en", "de", "es", "fr", "ja", "ko", "zh"]
                acc = {}
                final_output = ""
                for k in final_res:
                    if k.find("-") == -1:
                        continue
                    mode, lang = k.split("-")
                    if mode == "test":
                        # final_output += lang + " " + str(final_res[k]["acc"]) + '\t'
                        acc[lang] = round(final_res[k]["acc"] * 100, 2)

                for lang in order:
                    res_terms.append(acc[lang])
                res_terms.append(round(final_res["test_avg"]["acc"] * 100, 2))
                res_terms.append(round(final_res["valid_avg"]["acc"] * 100, 2))
                for term in res_terms:
                    final_output += str(term) + ","
                print(final_output[:-1])
            elif args.task_name == "panx":
                # order = ["ar", "he", "vi", "id", "jv", "ms", "tl", "eu", "ml", "ta", "te", "af", "nl", "en", "de", "el",
                #          "bn", "hi", "mr", "ur", "fa", "fr", "it", "pt", "es", "bg", "ru", "ja", "ka", "ko", "th", "sw",
                #          "yo", "my", "zh", "kk", "tr", "et", "fi", "hu"]
                order = ["en", "af", "ar", "bg", "bn", "de", "el", "es", "et", "eu", "fa", "fi", "fr", "he", "hi", "hu",
                         "id", "it", "ja", "jv", "ka", "kk", "ko", "ml", "mr", "ms", "my", "nl", "pt", "ru", "sw", "ta",
                         "te", "th", "tl", "tr", "ur", "vi", "yo", "zh"]
                f1 = {}
                final_output = ""
                for k in final_res:
                    if k.find("_") == -1:
                        continue
                    mode, lang = k.split("_")
                    if mode == "test":
                        f1[lang] = round(final_res[k]["f1"] * 100, 1)
                for lang in order:
                    res_terms.append(f1[lang])
                res_terms.append(round(final_res["test_avg"]["f1"] * 100, 1))
                res_terms.append(round(final_res["dev_avg"]["f1"] * 100, 1))
                for term in res_terms:
                    final_output += str(term) + ","
                print(final_output[:-1])
            elif args.task_name == "udpos":
                order = ["af", "ar", "bg", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "he", "hi", "hu", "id",
                         "it", "ja", "kk", "ko", "mr", "nl", "pt", "ru", "ta", "te", "th", "tl", "tr", "ur", "vi", "yo",
                         "zh"]
                f1 = {}
                final_output = ""
                for k in final_res:
                    if k.find("_") == -1:
                        continue
                    mode, lang = k.split("_")
                    if mode == "test":
                        f1[lang] = round(final_res[k]["f1"] * 100, 1)
                gap_sum = 0.0
                for lang in order:
                    if lang in f1:
                        res_terms.append(f1[lang])
                        gap_sum += f1["en"] - f1[lang]
                res_terms.append(round(final_res["test_avg"]["f1"] * 100, 1))
                res_terms.append(round(final_res["dev_avg"]["f1"] * 100, 1))

                res_terms.append(round(gap_sum / (len(res_terms) - 3), 1))

                for term in res_terms:
                    final_output += str(term) + ","
                print(final_output[:-1])
            elif args.task_name == "mlqa":
                # order = ["en", "es", "de", "ar", "hi", "vi", "zh"]
                order = ["en", "ar", "de", "es", "hi", "vi", "zh"]
                res = {}
                final_output = ""

                for k in final_res:
                    if k.find("_") == -1:
                        continue
                    mode, lang = k.split("_")
                    if mode == "test":
                        res[lang] = (round(final_res[k]["f1"], 1), round(final_res[k]["exact_match"], 1))
                gap_sum_f1, gap_sum_em = 0, 0
                for lang in order:
                    res_terms.append((res[lang][0], res[lang][1]))
                    gap_sum_f1 += res["en"][0] - res[lang][0]
                    gap_sum_em += res["en"][1] - res[lang][1]

                res_terms.append(
                    (round(final_res["test_avg"]["f1"], 1), round(final_res["test_avg"]["exact_match"], 1)))

                res_terms.append(
                    (round(final_res["dev_avg"]["f1"], 1), round(final_res["dev_avg"]["exact_match"], 1)))

                res_terms.append(
                    (round(gap_sum_f1 / (len(res_terms) - 3), 2), round(gap_sum_em / (len(res_terms) - 3), 2)))
                for term in res_terms:
                    final_output += str(term[0]) + '/' + str(term[1]) + ','
                print(final_output[:-1])
            elif args.task_name == "xquad":
                # order = ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]
                order = ["en", "ar", "de", "el", "es", "hi", "ru", "th", "tr", "vi", "zh"]

                res = {}
                final_output = ""

                for k in final_res:
                    if k.find("_") == -1:
                        continue
                    mode, lang = k.split("_")
                    if mode == "dev":
                        res[lang] = (round(final_res[k]["f1"], 2), round(final_res[k]["exact_match"], 2))
                for lang in order:
                    res_terms.append((res[lang][0], res[lang][1]))
                res_terms.append(
                    (round(final_res["dev_avg"]["f1"], 2), round(final_res["dev_avg"]["exact_match"], 2)))

                for term in res_terms:
                    final_output += str(term[0]) + '/' + str(term[1]) + ','
                print(final_output[:-1])
            elif args.task_name == "tydiqa":
                order = ["en", "ar", "bn", "fi", "id", "ko", "ru", "sw", "te"]
                res = {}
                final_output = ""

                for k in final_res:
                    if k.find("_") == -1:
                        continue
                    mode, lang = k.split("_")
                    if mode == "dev":
                        res[lang] = (round(final_res[k]["f1"], 2), round(final_res[k]["exact_match"], 2))
                for lang in order:
                    res_terms.append((res[lang][0], res[lang][1]))
                res_terms.append(
                    (round(final_res["dev_avg"]["f1"], 2), round(final_res["dev_avg"]["exact_match"], 2)))

                for term in res_terms:
                    final_output += str(term[0]) + '/' + str(term[1]) + ','
                print(final_output[:-1])
            elif args.task_name == "squad":
                res_terms.append(
                    (round(final_res["dev_en"]["f1"], 2), round(final_res["dev_en"]["exact_match"], 2)))
                final_output = ""
                for term in res_terms:
                    final_output += str(term[0]) + '/' + str(term[1]) + ','
                print(final_output[:-1])
                pass
            all_res_terms.append(res_terms)
        except:
            print()
            pass

    print(order)

    print(all_final_res)

    final_res_terms = []
    for i in range(len(all_res_terms[0])):
        s = 0.0 if args.task_name in ['xnli', "panx", "udpos", "pawsx"] else [0.0, 0.0]
        for j in range(len(all_res_terms)):
            if args.task_name in ['xnli', "panx", "udpos", "pawsx"]:
                s += all_res_terms[j][i]
            else:
                s[0] += all_res_terms[j][i][0]
                s[1] += all_res_terms[j][i][1]
        final_res_terms.append(
            round(s / len(all_res_terms), 2) if args.task_name in ['xnli', "panx", "udpos", "pawsx"] else (
                round(s[0] / len(all_res_terms), 2), round(s[1] / len(all_res_terms), 2)))

    final_output = '{}-average: '.format(len(all_res_terms))
    if args.task_name in ['xnli', "panx", "udpos", "pawsx"]:
        for term in final_res_terms:
            final_output += str(term) + ","
        print(final_output[:-1])
    else:
        for term in final_res_terms:
            final_output += str(term[0]) + '/' + str(term[1]) + ','
        print(final_output[:-1])
