from datasets import load_from_disk, load_dataset, Dataset
import pandas as pd


def get_merge_data():
    klue_dataset = load_from_disk("../input/data/train_dataset/")
    korquad_dataset = load_dataset("squad_kor_v1")

    dataset_dict = {}

    for i in ['train', 'validation']:
        merge_df = pd.concat([ pd.DataFrame(klue_dataset[i]), pd.DataFrame(korquad_dataset[i])], ignore_index=True)
        dataset_dict[i] = Dataset.from_pandas(merge_df)

    return dataset_dict