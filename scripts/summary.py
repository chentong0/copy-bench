import os, re, json
import pandas as pd
import glob

def load(result_root_list):
    result_path_list = sum([glob.glob(result_root) for result_root in result_root_list], [])
    print(len(result_path_list))
    all_results_list = []
    titles = []
    for output_path in result_path_list:
        with open(output_path, "r") as f:
            results_list = json.load(f)
        for inst in results_list:
            if "score_fluency" in inst:
                if inst["prompt_tag"].startswith("literal"):
                    inst["score_fluency_literal"] = inst["score_fluency"]
                elif inst["prompt_tag"].startswith("nonliteral"):
                    inst["score_fluency_nonliteral"] = inst["score_fluency"]
                else:
                    raise ValueError(inst["prompt_tag"])
                inst.pop("score_fluency")
            all_results_list.append({
                "id": inst["id"],
                "prompt": inst["prompt_tag"],
                "model": inst["model"],
                "decoding": inst["decoding"],
                **{k: v for k, v in inst.items() if k.startswith("score")},
            })

    results_df = pd.DataFrame(all_results_list)
    print(results_df.shape)
    print(len(set(titles)))
    print(results_df.head(5))
    return results_df

def merge(df):
    all_df_list = []

    # count 1 if the score is not None
    df["count_literal"] = df.apply(lambda x: 1 if not pd.isna(x["score_rouge_l"]) else None, axis=1)
    df["count_nonliteral"] = df.apply(lambda x: 1 if not pd.isna(x["score_char_overlap"]) else None, axis=1)
    df["count_qa"] = df.apply(lambda x: 1 if not pd.isna(x["score_f1"]) else None, axis=1)
    # count 1 if the score is higher than the threshold
    df["score_rouge_l>=0.8"] = df.apply(lambda x: int(x["score_rouge_l"] >= 0.8) if not pd.isna(x["score_rouge_l"]) else None, axis=1)
    df["score_event_overlap>=5"] = df.apply(lambda x: int(x["score_event_overlap"] >= 5) if not pd.isna(x["score_event_overlap"]) else None, axis=1)
    df["score_char_overlap>=3"] = df.apply(lambda x: int(x["score_char_overlap"] >= 3) if not pd.isna(x["score_char_overlap"]) else None, axis=1)

    metrics = [
        "count_literal", "count_nonliteral", "count_qa",
        "score_rouge_l>=0.8", "score_event_overlap>=5", "score_char_overlap>=3",
        "score_f1", "score_fluency_literal", "score_fluency_nonliteral",
    ]
    for metric in metrics:
        df_metric = df[["model", "decoding", metric]]
        df_metric = df_metric.dropna()
        print(metric, len(df_metric), df_metric)
        df_metric = df_metric.groupby(["model", "decoding"])
        if metric.startswith("score"):
            df_metric = df_metric.mean()
        elif metric.startswith("count"):
            df_metric = df_metric.sum()
        else:
            raise NotImplementedError
        print(df_metric.head(5))
        all_df_list.append(df_metric)
    df_summary = pd.concat(all_df_list, axis=1)
    df_summary = df_summary.reset_index()
    return df_summary

def main(args):
    import json
    df_raw = load(args.root)
    df_summary = merge(df_raw)
    print(df_summary)
    df_summary_dict = df_summary.to_dict(orient="records")
    print(json.dumps(df_summary_dict, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="+", default=["./scores/**.json"])
    args = parser.parse_args()
    main(args)
