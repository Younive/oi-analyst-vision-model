import json
from datasets import Dataset

def convert_to_conversation_format(example):
    """Convert each example to conversation messages format"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example['image_path']},
                {"type": "text", "text": example['user_input']},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": example['model_output']}
            ],
        },
    ]
    return {"messages": conversation}

def get_datasets():
    """Load and prepare datasets for training and evaluation"""
    # open dataset.json
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        dataset = Dataset.from_list(data)
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset columns: {dataset.column_names}")

        # format dataset to conversation messages
        formatted_data = [convert_to_conversation_format(ex) for ex in data]

        dataset = Dataset.from_list(formatted_data)
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset columns: {dataset.column_names}")

        # split dataset into train and eval
        if len(dataset) > 10:
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']
            print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = None
            print("Dataset too small for split, using all for training")
        return train_dataset, eval_dataset
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    