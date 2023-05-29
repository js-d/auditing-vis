from assess_utils.helpers import get_anomaly, img_folder_name2path, results_path
from assess_utils.lists import (
    methods,
    anomalous_models,
    null_models,
    lpips_nets,
    anomaly_types,
    localization_models,
)
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def max_null_score(lpips_net, method):
    max_score = 0
    for null_model in null_models:
        score = avg_anom_score(lpips_net, method, null_model)
        if score > max_score:
            max_score = score
    return max_score


def avg_anom_score(lpips_net, method, model_name):
    path = results_path / Path(
        "anomaly_scores", model_name, "detection_imagenet", f"{lpips_net}.json",
    )
    method_dct = json.load(open(path))[method]
    res = sum([method_dct[str(img_idx)] for img_idx in range(50)]) / 50
    return res


def agg_max_null_score(lpips_net):
    car_methods = [m for m in methods if m.startswith("car")]
    arr_null_scores = np.zeros((len(null_models), len(car_methods), 50))
    for i, null_model in enumerate(null_models):
        path = results_path / Path(
            "anomaly_scores", null_model, "detection_imagenet", f"{lpips_net}.json",
        )
        for j, method in enumerate(car_methods):
            method_dct = json.load(open(path))[method]
            arr_null_scores[i, j, :] = [
                method_dct[str(img_idx)] for img_idx in range(50)
            ]
    avg_null_scores = np.mean(arr_null_scores, axis=(1, 2))
    res = np.max(avg_null_scores)
    return res


def detect_detailed_res():
    dct = {"lpips_net": [], "method": [], "anomaly": [], "anom_score_ratio": []}
    for lpips_net in lpips_nets:
        for method in methods:
            max_null = max_null_score(lpips_net, method)
            for model_name in anomalous_models:
                anom_score = avg_anom_score(lpips_net, method, model_name)
                ratio = anom_score / max_null
                print(lpips_net, method, model_name, ratio)
                dct["lpips_net"].append(lpips_net)
                dct["method"].append(method)
                dct["anomaly"].append(model_name)
                dct["anom_score_ratio"].append(ratio)

    # get score for all caricatures by averaging their anomaly scores
    car_methods = [m for m in methods if m.startswith("car")]
    for lpips_net in lpips_nets:
        # get max null anom score for "caricature"
        max_null = agg_max_null_score(lpips_net)

        # get avg anom scores for "caricature"
        for model_name in anomalous_models:
            anom_score = sum(
                [
                    avg_anom_score(lpips_net, method, model_name)
                    for method in car_methods
                ]
            ) / len(car_methods)
            ratio = anom_score / max_null
            print(lpips_net, "caricatures", model_name, ratio)
            dct["lpips_net"].append(lpips_net)
            dct["method"].append("caricatures")
            dct["anomaly"].append(model_name)
            dct["anom_score_ratio"].append(ratio)
    df = pd.DataFrame(dct)
    save_path = results_path / Path("detection_res", "detailed.csv")
    df.to_csv(save_path)
    return df


def detect_frac_res(detailed_df):
    dct = {"lpips_net": [], "method": [], "anom_type": [], "detect_frac": []}
    for lpips_net in lpips_nets:
        for method in sorted(methods + ["caricatures"]):
            for anom_type in anomaly_types:
                select_df = detailed_df[
                    (detailed_df["lpips_net"] == lpips_net)
                    & (detailed_df["method"] == method)
                    & (detailed_df["anomaly"].str.startswith(anom_type))
                ]
                detect_frac = np.mean(select_df["anom_score_ratio"] > 1)
                print(lpips_net, method, anom_type, detect_frac)
                dct["lpips_net"].append(lpips_net)
                dct["method"].append(method)
                dct["anom_type"].append(anom_type)
                dct["detect_frac"].append(detect_frac)
    df = pd.DataFrame(dct)
    save_path = results_path / Path("detection_res", "detect_frac.csv")
    df.to_csv(save_path)
    return df


def localization_res(img_idx_start=0, img_idx_end=40):
    dct = {"lpips_net": [], "method": [], "anomaly": [], "auc": []}
    for lpips_net in lpips_nets:
        for model_name in localization_models:
            normal_path = results_path / Path(
                "anomaly_scores",
                model_name,
                f"localization_{model_name}_normal",
                f"{lpips_net}.json",
            )
            abnormal_path = results_path / Path(
                "anomaly_scores",
                model_name,
                f"localization_{model_name}_abnormal",
                f"{lpips_net}.json",
            )
            normal_dct = json.load(open(normal_path))
            abnormal_dct = json.load(open(abnormal_path))
            for method in methods:
                print(lpips_net, method, model_name)
                normal_scores = [
                    normal_dct[method][str(img_idx)]
                    for img_idx in range(img_idx_start, img_idx_end)
                ]
                abnormal_scores = [
                    abnormal_dct[method][str(img_idx)]
                    for img_idx in range(img_idx_start, img_idx_end)
                ]
                y_true = len(normal_scores) * [0] + len(abnormal_scores) * [1]
                y_scores = normal_scores + abnormal_scores
                auc = roc_auc_score(y_true, y_scores)
                # print("normal_scores", normal_scores)
                # print("abnormal_scores", abnormal_scores)
                print("auc", auc)
                print()
                dct["lpips_net"].append(lpips_net)
                dct["method"].append(method)
                dct["anomaly"].append(model_name)
                dct["auc"].append(auc)
    df = pd.DataFrame(dct)
    save_path = results_path / Path("localization_res", "new_table.csv")
    df.to_csv(save_path)
    return df


if __name__ == "__main__":
    print("detection")
    detailed_df = detect_detailed_res()
    detect_frac_df = detect_frac_res(detailed_df)

    print("localization")
    df = localization_res(0, 40)
