import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
 
# Function to clear CUDA cache
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA cache cleared")
 
# Clear cache at the beginning
clear_cuda_cache()
 
# Check CUDA availability and GPU count
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
else:
    print("CUDA is not available. Using CPU.")
    num_gpus = 0
 
# Disable gradient computation
torch.set_grad_enabled(False)
 
# Load the model
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, torch_dtype=torch.float16)
 
# Use DataParallel for multiple GPUs
if num_gpus > 1:
    model = torch.nn.DataParallel(model)
 
model = model.to(device='cuda')
model.eval()  # Set the model to evaluation mode
 
# Disable gradient computation for model parameters
for param in model.parameters():
    param.requires_grad = False
 
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
 
# Clear cache after model loading
clear_cuda_cache()
 
image = Image.open("C:/Users/renot/Downloads/Image (1).jpg").convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]
 
# Use torch.no_grad() to ensure no gradients are computed or stored
with torch.no_grad():
    res = model.module.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
    )
print(res)
 
# Clear cache after inference
clear_cuda_cache()
 
# # Streaming part
# with torch.no_grad():
#     res = model.module.chat(
#         image=image,
#         msgs=msgs,
#         tokenizer=tokenizer,
#         sampling=True,
#         temperature=0.7,
#         stream=True
#     )
 
#     generated_text = ""
#     for new_text in res:
#         generated_text += new_text
#         print(new_text, flush=True, end='')
 
# # Clear cache at the end
# clear_cuda_cache()