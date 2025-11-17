import pandas as pd
import os 
import numpy as np

def read_json(path):
    return pd.read_json(path, lines=True)

def stat_data(df, column="proficiency_level"):
    stats = df[column].value_counts().to_dict()
    total = len(df)
    for key in stats:
        stats[key] = {
            "count": stats[key],
            "percentage": stats[key] / total * 100
        }
    return stats

def split_data(df):
    # Split the data into training and validation sets
    val_df = pd.DataFrame()
    l = len(df)
    stats = stat_data(df, column="proficiency_level")
    for col in df["proficiency_level"].unique():
        subset = df[df["proficiency_level"] == col]
        val_df = pd.concat([val_df, subset.sample(frac=.25 * 25 / stats[col]["percentage"], random_state=42)])
    train_df = df.drop(val_df.index)
    return train_df, val_df

def save_json(df, path):
    df.to_json(path, lines=True, orient="records")

if __name__ == "__main__":
    train_path = '/home/tellong/Bureau/Code/Code/GazeSkill/SkillFormer/annotations/annotations_train.jsonl'
    val_path = '/home/tellong/Bureau/Code/Code/GazeSkill/SkillFormer/annotations/annotations_val.jsonl'
    
    train_df = read_json(train_path)
    val_df = read_json(val_path)
    data = pd.concat([train_df, val_df], ignore_index=True)
    
    print("Original train dataset statistics:", stat_data(train_df))
    print("Original validation dataset statistics:", stat_data(val_df))
    
    train_df, val_df = split_data(data) 

    print("Train dataset statistics:", stat_data(train_df))
    print("Validation dataset statistics:", stat_data(val_df))
    
    save_json(train_df, os.path.join(os.path.dirname(train_path), "balanced_annotations_train.jsonl"))
    save_json(val_df, os.path.join(os.path.dirname(val_path), "balanced_annotations_val.jsonl"))