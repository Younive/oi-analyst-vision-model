from transformers import AutoProcessor
from unsloth import FastVisionModel
import torch
from PIL import Image

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_processor():
    try:
        # loading model
        print("Loading model...")
        model, _ = FastVisionModel.from_pretrained(
            "oi_analyst_model",
            load_in_4bit=True,
            trust_remote_code=True,
        )
        # Switch to inference mode
        FastVisionModel.for_inference(model)

        # Load processor
        processor = AutoProcessor.from_pretrained("oi_analyst_model")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    return model, processor

def analyze_chart(image_path, instruction="Analyze this Intraday Volume chart."):
    """
    Generate chart analysis from an image
    
    Args:
        image_path: Path to the chart image
        instruction: User instruction/prompt
    
    Returns:
        Generated analysis text
    """

    # load model, processor
    model, processor = load_model_and_processor()
    
    # load the actual image file
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image.size}")
    except Exception as e:
        return f"Error loading image: {e}"
    
    # Create conversation messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    # Apply chat template (creates formatted text with <image> tokens)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"Formatted prompt (first 100 chars): {text[:100]}...")
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,     
            temperature=0.7,          
            do_sample=True,           
            top_p=0.9,               
            repetition_penalty=1.1, 
        )
    
    # Decode output
    full_response = processor.decode(outputs[0], skip_special_tokens=True)
    

    if "<|im_start|>assistant" in full_response:
        # Split at assistant token and take the last part
        response = full_response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
    else:
        # Fallback: remove the input text
        response = full_response.replace(text, "").strip()
    
    return response

if __name__ == "__main__":

    # Test image path
    test_image_path = "test_oi_gold.jpg" 
    
    # Generate analysis
    analysis = analyze_chart(
        test_image_path,
        instruction="วิเคราะห์กราฟ Options นี้โดยละเอียด"
    )
    
    print("\n" + "="*60)
    print("GENERATED ANALYSIS:")
    print("="*60)
    print(analysis)
    print("="*60 + "\n")