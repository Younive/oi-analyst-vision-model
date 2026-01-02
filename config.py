import random

IMAGES_FOLDER = 'images'
TEXTS_FOLDER = 'texts'
OUTPUT_FILE = 'dataset.json'
instructions = [
    'Analyze this Intraday Volume chart. Provide pattern, strategy, and volatility.',
    'วิเคราะห์กราฟ Options นี้โดยละเอียด',
    'Explain the Put/Call distribution and suggest trading strategies.',
    'อธิบายกราฟและให้คำแนะนำการเทรด',
]
USER_INSTRUCTION = random.choice(instructions)

BASE_MODEL = 'unsloth/Qwen3-VL-4B-Instruct'
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
EPOCHS = 5

