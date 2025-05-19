from transformers import Emu3Processor, Emu3ImageProcessor
import torch
               

processor = Emu3Processor.from_pretrained(
     "BAAI/Emu3-Gen-hf",
      # torch_dtype=torch.bfloat16,
      device_map="balanced",
      )
image_processor = Emu3ImageProcessor(
    min_pixels=256 * 256,
    max_pixels=256 * 256,
    # torch_dtype=torch.bfloat16
    device_map="balanced",
    )
tokenizer = processor.tokenizer

image_token = tokenizer.image_token
image_start_token = (
    tokenizer.boi_token
)  # "<|image start|>" fixed tokens for start and end of image
image_end_token = tokenizer.eoi_token  # "<|image end|>"
fake_token_around_image = (
    tokenizer.image_wrapper_token
)  # "<|image token|>"  every image starts with it
eof_token = tokenizer.eof_token  # "<|extra_201|>"
bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token
downsample_ratio = 8


def add_bos_eos(string):
    for i, word in enumerate(string.split(" ")):
        if i == 0:
            new_string = f"{bos_token}{word}"
        elif word in ["USER:", "ASSISTANT:"]:
            new_string = new_string.strip() + f"{eos_token}"
            new_string += f"{bos_token}{word}"
        else:
            new_string += f" {word}"
    new_string = new_string.strip()
    eval_mode = True
    if not eval_mode:
        new_string += f"{eos_token}"
    return new_string
def preprocess_conversation(text, image_features):
    # start_sentence = "<|extra 203|>"
    eof_token = "<|extra_201|>"
    image_start_token = "<|image start|>"
    image_end_tokens= "<|image end|>",
    image_sizes = iter(image_features.image_sizes)
    image_start_tokens = f"{image_start_token}"
    image_end_tokens = f"{eof_token}{image_end_token}"

    prompt_strings = []
    for sample in text:
        # here we add end of turn tokens useful for chat finetuning
        sample = add_bos_eos(sample)

        while image_token in sample:
            image_size = next(image_sizes)
            height, width = image_size
            height = height // downsample_ratio
            width = width // downsample_ratio
            image_seq_length = height * (
                width + 1
            )  # +1 for extra row when converting to BPE in modeling code

            image_placeholder = f"{image_start_tokens}{height}*{width}{fake_token_around_image}{'<placeholder>' * image_seq_length}{image_end_tokens}"
            sample = sample.replace(image_token, image_placeholder, 1)

        prompt_strings.append(sample)

    text = [sample.replace("<placeholder>", image_token) for sample in prompt_strings]

    return text

# def prefix_allowed_tokens_fn(batch_id, input_ids):
#     height, width = HEIGHT, WIDTH
#     visual_tokens = VISUAL_TOKENS
#     image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=model.device)
#     eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
#     eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
#     pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=model.device)
#     eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=model.device)
#     eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

#     position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
#     offset = input_ids.shape[0] - position
#     if offset % (width + 1) == 0:
#         return (eol_token_id,)
#     elif offset == (width + 1) * height + 1:
#         return (eof_token_id,)
#     elif offset == (width + 1) * height + 2:
#         return (eoi_token_id,)
#     elif offset == (width + 1) * height + 3:
#         return (eos_token_id,)
#     elif offset > (width + 1) * height + 3:
#         return (pad_token_id,)
#     else:
#         return visual_tokens