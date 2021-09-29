import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--translation_path",
        default=None,
        type=str,
        required=True,
        help="",
    )

    drop_languages = ["en", "zh-CN", "zh", "ja", "ko", "th", "my", "ml", "ta"]
    translate_languages = None
    args = parser.parse_args()
    src2tgt = {}
    print("Reading translation from {}".format(args.translation_path))
    with open(args.translation_path, encoding="utf-8") as f:
        cnt = 0
        for line in f:
            cnt += 1
            if cnt % 10000 == 0:
                print("Reading lines {}".format(cnt))
            items = line.split("\t")

            if items == 3:
                src_sent, tgt_lang, tgt_sent = line.split("\t")
                alignment = None
            else:
                src_sent, tgt_lang, tgt_sent, alignment_str = line.split("\t")
                alignment = []
                for x in alignment_str.split(" "):
                    alignment.append((int(x.split("/")[0]), int(x.split("/")[1])))

            if tgt_lang in drop_languages:
                continue
            if translate_languages is not None and tgt_lang not in translate_languages:
                continue

            cnt_src = {}
            cnt_tgt = {}
            for x in alignment:

                if x[0] not in cnt_src:
                    cnt_src[x[0]] = 0
                cnt_src[x[0]] += 1

                if x[1] not in cnt_tgt:
                    cnt_tgt[x[1]] = 0
                cnt_tgt[x[1]] += 1

                if not (cnt_src[x[0]] <= 1 or cnt_tgt[x[1]] <= 1):
                    print(cnt_src, cnt_tgt)
                    print(alignment)
                    print(src_sent, tgt_sent)

                assert cnt_src[x[0]] <= 1 or cnt_tgt[x[1]] <= 1


