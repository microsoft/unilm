import os
import sys
import constants


def page_hits_level_metric(
        vertical,
        target_website,
        sub_output_dir,
        prev_voted_lines
):
    """Evaluates the hit level prediction result with precision/recall/f1."""

    all_precisions = []
    all_recall = []
    all_f1 = []

    lines = prev_voted_lines

    evaluation_dict = dict()

    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        html_path = items[0]
        text = items[2]
        truth = items[3]  # gt for this node
        pred = items[4]  # pred-value for this node
        if truth not in evaluation_dict and truth != "none":
            evaluation_dict[truth] = dict()
        if pred not in evaluation_dict and pred != "none":
            evaluation_dict[pred] = dict()
        if truth != "none":
            if html_path not in evaluation_dict[truth]:
                evaluation_dict[truth][html_path] = {"truth": set(), "pred": set()}
            evaluation_dict[truth][html_path]["truth"].add(text)
        if pred != "none":
            if html_path not in evaluation_dict[pred]:
                evaluation_dict[pred][html_path] = {"truth": set(), "pred": set()}
            evaluation_dict[pred][html_path]["pred"].add(text)
    metric_str = "tag, num_truth, num_pred, precision, recall, f1\n"
    for tag in evaluation_dict:
        num_html_pages_with_truth = 0
        num_html_pages_with_pred = 0
        num_html_pages_with_correct = 0
        for html_path in evaluation_dict[tag]:
            result = evaluation_dict[tag][html_path]
            if result["truth"]:
                num_html_pages_with_truth += 1
            if result["pred"]:
                num_html_pages_with_pred += 1
            if result["truth"] & result["pred"]: # 似乎这里是个交集...不能随便乱搞
                num_html_pages_with_correct += 1

        precision = num_html_pages_with_correct / (
                num_html_pages_with_pred + 0.000001)
        recall = num_html_pages_with_correct / (
                num_html_pages_with_truth + 0.000001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.000001)
        metric_str += "%s, %d, %d, %.2f, %.2f, %.2f\n" % (
            tag, num_html_pages_with_truth, num_html_pages_with_pred, precision,
            recall, f1)
        all_precisions.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    output_path = os.path.join(sub_output_dir, "scores", f"{target_website}-final-scores.txt")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write(metric_str)
        print(f.name, file=sys.stderr)
    print(metric_str, file=sys.stderr)
    return sum(all_precisions) / len(all_precisions), sum(all_recall) / len(all_recall), sum(all_f1) / len(all_f1)


def site_level_voting(vertical, target_website, sub_output_dir, prev_voted_lines):
    """Adds the majority voting for the predictions."""

    lines = prev_voted_lines

    field_xpath_freq_dict = dict()

    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        xpath = items[1]
        pred = items[4]
        if pred == "none":
            continue
        if pred not in field_xpath_freq_dict:
            field_xpath_freq_dict[pred] = dict()
        if xpath not in field_xpath_freq_dict[pred]:
            field_xpath_freq_dict[pred][xpath] = 0
        field_xpath_freq_dict[pred][xpath] += 1

    most_frequent_xpaths = dict()  # Site level voting.
    for field, xpth_freq in field_xpath_freq_dict.items():
        frequent_xpath = sorted(
            xpth_freq.items(), key=lambda kv: kv[1], reverse=True)[0][0]  # Top 1.
        most_frequent_xpaths[field] = frequent_xpath

    voted_lines = []
    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        xpath = items[1]
        flag = "none"
        for field, most_freq_xpath in most_frequent_xpaths.items():
            if xpath == most_freq_xpath:
                flag = field
        if items[4] == "none" and flag != "none":
            items[4] = flag
        voted_lines.append("\t".join(items))

    output_path = os.path.join(sub_output_dir, "preds", f"{target_website}-final-preds.txt")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write("\n".join(voted_lines))

    return page_hits_level_metric(  # re-eval with the voted prediction
        vertical,
        target_website,
        sub_output_dir,
        voted_lines
    )


def page_level_constraint(vertical, target_website,
                          lines, sub_output_dir):
    """Takes the top highest prediction for empty field by ranking raw scores."""
    """
    In this step, we make sure every node has a prediction
    """

    tags = constants.ATTRIBUTES_PLUS_NONE[vertical]

    site_field_truth_exist = dict()
    page_field_max = dict()
    page_field_pred_count = dict()
    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        html_path = items[0]
        truth = items[3]
        pred = items[4]
        if pred != "none":
            if pred not in page_field_pred_count:
                page_field_pred_count[pred] = 0
            page_field_pred_count[pred] += 1
            continue
        raw_scores = [float(x) for x in items[5].split(",")]
        assert len(raw_scores) == len(tags)
        site_field_truth_exist[truth] = True
        for index, score in enumerate(raw_scores):
            if html_path not in page_field_max:
                page_field_max[html_path] = {}
            if tags[index] not in page_field_max[
                html_path] or score >= page_field_max[html_path][tags[index]]:
                page_field_max[html_path][tags[index]] = score
    print(page_field_pred_count, file=sys.stderr)
    voted_lines = []
    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        html_path = items[0]
        raw_scores = [float(x) for x in items[5].split(",")]
        pred = items[4]
        for index, tag in enumerate(tags):
            if tag in site_field_truth_exist and tag not in page_field_pred_count:
                if pred != "none":
                    continue
                if raw_scores[index] >= page_field_max[html_path][tags[index]] - (1e-3):
                    items[4] = tag
        voted_lines.append("\t".join(items))

    return site_level_voting(
        vertical, target_website, sub_output_dir, voted_lines)
