import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM


def load_tokenizer(tokenizer_path: str) -> BPETokenizer:
    return BPETokenizer.load(tokenizer_path)


def load_model(model_path: str, tokenizer: BPETokenizer, device: torch.device) -> TinyStoriesForCausalLM:
    # Load config from args.json if exists (same as generate_tinystories_text.py)
    config_path = os.path.join(os.path.dirname(model_path), "args.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)
        config = TinyStoriesConfig(
            vocab_size=len(tokenizer.token2id),
            hidden_size=train_args.get("hidden_size", 256),
            num_hidden_layers=train_args.get("num_layers", 4),
            num_attention_heads=train_args.get("num_heads", 8),
            intermediate_size=train_args.get("intermediate_size", 1024),
            hidden_dropout_prob=train_args.get("dropout", 0.1),
            attention_probs_dropout_prob=train_args.get("dropout", 0.1),
            max_position_embeddings=train_args.get("max_seq_len", 512),
            window_size=train_args.get("window_size", 256),
        )
    else:
        config = TinyStoriesConfig(vocab_size=len(tokenizer.token2id))

    model = TinyStoriesForCausalLM(config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Mac MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_prompts(prompt_file: str | None) -> list[str]:
    if prompt_file is None:
        raise ValueError("Provide --prompt_file (txt). One prompt per line.")
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                prompts.append(p)
    if not prompts:
        raise ValueError("Prompt file is empty.")
    return prompts


@torch.no_grad()
def generate_one(
    model: TinyStoriesForCausalLM,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict:
    # Encode prompt
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=True)],
        dtype=torch.long,
        device=device,
    )
    input_len = input_ids.shape[1]

    eos_token_id = tokenizer.token2id.get("<eos>", None)

    # Generate (IMPORTANT: output contains prompt + generated)
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )

    # Slice out ONLY the generated continuation (fixes your issue)
    gen_ids = output_ids[0][input_len:]
    generated_text = tokenizer.decode(gen_ids.tolist())

    # (Optional) keep full text for debugging
    full_text = tokenizer.decode(output_ids[0].tolist())

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate multiple stories for diversity/stability evaluation.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl")
    parser.add_argument("--prompt_file", type=str, required=True, help="txt file, one prompt per line")
    parser.add_argument("--output_jsonl", type=str, default="outputs/diversity_generations.jsonl")

    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.2, 0.7, 1.0, 1.3])
    parser.add_argument("--samples_per_prompt", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=160)

    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)

    # if you don't want to store full_text to save space
    parser.add_argument("--save_full_text", action="store_true")

    args = parser.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)

    Path(os.path.dirname(args.output_jsonl) or ".").mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(args.model_path, tokenizer, device)

    prompts = read_prompts(args.prompt_file)

    total = len(prompts) * len(args.temperatures) * args.samples_per_prompt
    print(f"Device: {device}")
    print(f"Prompts: {len(prompts)}, Temps: {args.temperatures}, Samples/prompt: {args.samples_per_prompt}")
    print(f"Total generations: {total}")
    print(f"Saving to: {args.output_jsonl}")

    out_path = Path(args.output_jsonl)
    # overwrite (safer for clean runs)
    if out_path.exists():
        out_path.unlink()

    idx = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt_id, prompt in enumerate(prompts):
            for temp in args.temperatures:
                for sample_id in range(args.samples_per_prompt):
                    idx += 1
                    ex = generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        device=device,
                        max_length=args.max_length,
                        temperature=float(temp),
                        top_k=args.top_k,
                        top_p=args.top_p,
                    )

                    row = {
                        "prompt_id": prompt_id,
                        "temperature": float(temp),
                        "sample_id": sample_id,
                        "prompt": ex["prompt"],
                        "generated_text": ex["generated_text"],
                    }
                    if args.save_full_text:
                        row["full_text"] = ex["full_text"]

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    if idx % 50 == 0:
                        print(f"  Progress: {idx}/{total}")

    print("Done.")


if __name__ == "__main__":
    main()