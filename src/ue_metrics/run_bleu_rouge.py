import pandas as pd
from truthfulqa.metrics import run_bleu_and_rouge


if __name__ == "__main__":
    df = pd.read_csv("../../data/results_seq/generation/truthfulqa_gen_ddg_google_qwen_70b.csv")
    result = run_bleu_and_rouge(model_key="text", frame=df)
