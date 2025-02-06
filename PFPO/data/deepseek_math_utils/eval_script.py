import regex
from copy import deepcopy
from data.deepseek_math_utils.eval_utils import math_equal
from data.deepseek_math_utils.ocwcourses_eval_utils import normalize_numeric, numeric_equality, normalize_symbolic_equation, SymbolicMathMixin


def is_correct(item, pred_key='prediction', prec=1e-3):
    pred = item[pred_key]
    ans = item['answer']
    if isinstance(pred, list) and isinstance(ans, list):
        pred_matched = set()
        ans_matched = set()
        for i in range(len(pred)):
            for j in range(len(ans)):
                item_cpy = deepcopy(item)
                item_cpy.update({
                    pred_key: pred[i],
                    'answer': ans[j]
                })
                if is_correct(item_cpy, pred_key=pred_key, prec=prec):
                    pred_matched.add(i)
                    ans_matched.add(j)
                    if item_cpy[pred_key] == '2,3,4':
                        print(item, flush=True)
                        print("wtf", flush=True)
        return len(pred_matched) == len(pred) and len(ans_matched) == len(ans)
    elif isinstance(pred, str) and isinstance(ans, str):
        if '\\cup' in pred and '\\cup' in ans:
            item = deepcopy(item)
            item.update({
                pred_key: pred.split('\\cup'),
                'answer': ans.split('\\cup'),
            })
            return is_correct(item, pred_key=pred_key, prec=prec)
        else:
            label = False
            try:
                label = abs(float(regex.sub(r',', '', str(pred))) - float(regex.sub(r',', '', str(ans)))) < prec
            except:
                pass
            label = label or (ans and pred == ans) or math_equal(pred, ans)
            return label
    else:
        print(item, flush=True)
        raise NotImplementedError()


def eval_math(item, pred_key='prediction', prec=1e-3):
    pred = item[pred_key]
    if pred_key == 'program_output' and isinstance(pred, str):
        pred = [pred]
    ans = item['answer']
    if isinstance(pred, list) and isinstance(ans, list):
        # for some questions in MATH, `reference` repeats answers
        _ans = []
        for a in ans:
            if a not in _ans:
                _ans.append(a)
        ans = _ans
        # some predictions for MATH questions also repeats answers
        _pred = []
        for a in pred:
            if a not in _pred:
                _pred.append(a)
        # some predictions mistakenly box non-answer strings
        pred = _pred[-len(ans):]

    item.update({
        pred_key: pred,
        'answer': ans
    })
    return is_correct(item, pred_key=pred_key, prec=prec)


def eval_last_single_answer(item, pred_key='prediction', prec=1e-3):
    for key in [pred_key, 'answer']:
        assert isinstance(item[key], str), f"{key} = `{item[key]}` is not a str"
    return is_correct(item, pred_key=pred_key, prec=prec)


def eval_agieval_gaokao_math_cloze(item, pred_key='prediction', prec=1e-3):
    if pred_key == 'program_output' and isinstance(item[pred_key], str):
        item[pred_key] = [item[pred_key]]
    for key in [pred_key, 'answer']:
        assert isinstance(item[key], list), f"{key} = `{item[key]}` is not a list"
    pred = item[pred_key]
    ans = item['answer']
    _pred = []
    for p in pred:
        p = p + ";"
        while p:
            left_brackets = 0
            for i in range(len(p)):
                if p[i] == ';' or (p[i] == ',' and left_brackets == 0):
                    _p, p = p[:i].strip(), p[i + 1:].strip()
                    if _p not in _pred:
                        _pred.append(_p)
                    break
                elif p[i] in '([{':
                    left_brackets += 1
                elif p[i] in ')]}':
                    left_brackets -= 1
    pred = _pred[-len(ans):]
    if len(pred) == len(ans):
        for p, a in zip(pred, ans):
            item.update({
                pred_key: p,
                'answer': a,
            })
            if not is_correct(item, pred_key=pred_key, prec=prec):
                return False
        return True
    else:
        return False


def eval_agieval_gaokao_mathqa(item, pred_key='prediction', prec=1e-3):
    if pred_key == 'program_output' and isinstance(item[pred_key], str):
        item[pred_key] = [item[pred_key]]
    pred_str = " ".join(item[pred_key])
    ans = item['answer']
    tag = None
    idx = -1
    for t in 'ABCD':
        if t in pred_str and pred_str.index(t) > idx:
            tag = t
            idx = pred_str.index(t)
    return tag == ans


def eval_math_sat(item, pred_key='prediction', prec=1e-3):
    for key in [pred_key, 'answer']:
        assert isinstance(item[key], str), f"{key} = `{item[key]}` is not a str"
    return item[pred_key].lower() == item['answer'].lower()


def eval_mmlu_stem(item, pred_key='prediction', prec=1e-3):
    return eval_math_sat(item, pred_key=pred_key, prec=prec)


def eval_ocwcourses(item, pred_key='prediction', prec=1e-3):
    INVALID_ANSWER = "[invalidanswer]"
    for key in [pred_key, 'answer']:
        assert isinstance(item[key], str), f"{key} = `{item[key]}` is not a str"
    pred = item[pred_key]
    ans = item['answer']

    try:
        float(ans)
        normalize_fn = normalize_numeric
        is_equiv = numeric_equality
        answer_type = "numeric"
    except ValueError:
        if "=" in ans:
            normalize_fn = normalize_symbolic_equation
            is_equiv = lambda x, y: x == y
            answer_type = "equation"
        else:
            normalize_fn = SymbolicMathMixin().normalize_tex
            is_equiv = SymbolicMathMixin().is_tex_equiv
            answer_type = "expression"

    correct_answer = normalize_fn(ans)

    unnormalized_answer = pred if pred else INVALID_ANSWER
    model_answer = normalize_fn(unnormalized_answer)

    if unnormalized_answer == INVALID_ANSWER:
        acc = 0
    elif model_answer == INVALID_ANSWER:
        acc = 0
    elif is_equiv(model_answer, correct_answer):
        acc = 1
    else:
        acc = 0

    return acc


def eval_minif2f_isabelle(item, pred_key='prediction', prec=1e-3):
    return True
