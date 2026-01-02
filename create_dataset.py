import os
import json
from config import IMAGES_FOLDER, TEXTS_FOLDER, OUTPUT_FILE, USER_INSTRUCTION

def create_dataset():
  dataset = []
  try:
    image_files = os.listdir(IMAGES_FOLDER)
    print(f"Found {len(image_files)} image files.")

    for image in image_files:

      base_name = os.path.splitext(image)[0]
      text_file = base_name+'.txt'
      text_path = os.path.join(TEXTS_FOLDER, text_file)

      if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
          analysis = f.read().strip()
        dataset.append({
            'image_path': os.path.join(os.getcwd(), IMAGES_FOLDER, image),
            'user_input': USER_INSTRUCTION,
            'model_output': analysis
        })
      else:
        print(f'warning: missing text for {image}')

      with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
  except Exception as e:
    print(f"An error occurred: {e}")
    raise

  print(f"Dataset created with {len(dataset)} entries.")

if __name__ == "__main__":
    create_dataset()