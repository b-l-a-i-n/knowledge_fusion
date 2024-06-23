import re
import pandas as pd
from sklearn.metrics import classification_report


def get_answers(text):
    matches = re.findall("\d+", text)
    return [int(i) for i in matches]


def preprocess(preds):
    new_preds = preds.copy()
    new_preds['raw_prediction'] = new_preds["prediction"]
    new_preds['prediction'] = new_preds["prediction"].apply(get_answers)
    new_preds['match_dict'] = new_preds['match_dict'].apply(eval)
    new_preds['prediction'] = [[x['match_dict'][j] for j in x['prediction'] if j in x['match_dict']] for i, x in new_preds.iterrows()]
    return new_preds


def parse(preds, df, greedy=True, report=False):
    preds = preprocess(preds)  
    new_df = {"sample_id": [], "prediction": []}
    for i, row in preds.iterrows():
        for item, id in row['match_dict'].items():
            new_df["sample_id"].append(id)
            if greedy:
                new_df["prediction"].append(id in row['prediction'][:1])
            else:
                new_df["prediction"].append(id in row['prediction'])

    result = pd.DataFrame(data=new_df)
    result = result.sort_values(by=["sample_id"])
    result.prediction = result.prediction.astype(np.int32)

    if report:
        pred_labels = result['prediction']
        df = df.sort_values(by=["sample_id"])
        true_labels = df['correct'].astype(np.int32).values
        print(classification_report(pred_labels, true_labels))
    
    return result


if __name__ == "__main__":
    df_file_path = "TextGraphs17-shared-task/data/tsv/test.tsv"
    pred_file_path = "llama_70B_test_ds_ws.csv"
    
    df = pd.read_csv(df_file_path, sep="\t")
    preds = pd.read_csv(pred_file_path, sep="\t")
    
    submit = parse(preds, df)
    submit.to_csv("submission.tsv", index=False, sep="\t")
