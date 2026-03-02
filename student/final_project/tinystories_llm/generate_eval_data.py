import sys
import os
import json
import torch
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())

from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    print("Loading tokenizer and model...")
    tokenizer = BPETokenizer.load(tokenizer_path)

    config_path = os.path.join(os.path.dirname(model_path), 'args.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_args = json.load(f)
        config = TinyStoriesConfig(
            vocab_size=len(tokenizer.token2id),
            hidden_size=train_args.get('hidden_size', 256),
            num_hidden_layers=train_args.get('num_layers', 4),
            num_attention_heads=train_args.get('num_heads', 8),
            max_position_embeddings=train_args.get('max_seq_len', 256),
            window_size=train_args.get('window_size', 256),
        )
    else:
        config = TinyStoriesConfig(vocab_size=len(tokenizer.token2id), max_position_embeddings=256)

    # Model Load
    model = TinyStoriesForCausalLM(config)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation texts at different temperatures.")
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--tokenizer_path', type=str, default='bpe_tokenizer_tinystories.pkl')
    parser.add_argument('--prompts_file', type=str, default='evaluation_prompts.json')
    parser.add_argument('--output_file', type=str, default='base_model_generated_results.json')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--device', type=str, default='mps') 
    
    # 🌟 새롭게 추가된 옵션들
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per prompt')
    args = parser.parse_args()

    # 모델 준비
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path, args.device)
    eos_token_id = tokenizer.token2id.get('<eos>', None)

    # 프롬프트 로드
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    # 🌟 온도 4개로 세분화
    temperatures = [0.1, 0.5, 0.8, 1.0]
    results = []

    print(f"Starting generation...")
    print(f"- Prompts: {len(prompts)}")
    print(f"- Temperatures: {temperatures}")
    print(f"- Samples per prompt: {args.num_samples}")
    
    # 총 진행률 바 설정 (30개 * 4온도 * 10샘플 = 총 1,200번)
    total_iterations = len(prompts) * len(temperatures) * args.num_samples
    
    with tqdm(total=total_iterations, desc="Generating Texts") as pbar:
        for temp in temperatures:
            for item in prompts:
                prompt_text = item["text"]
                input_ids = torch.tensor([tokenizer.encode(prompt_text, add_special_tokens=True)], dtype=torch.long).to(args.device)
                
                # 🌟 동일한 프롬프트에 대해 N번 반복 생성
                for sample_idx in range(args.num_samples):
                    # 모델이 똑같은 답만 내뱉지 않도록 시드(seed)를 매번 바꿔줄 수도 있지만,
                    # Temperature가 0.1보다 크면 자동으로 다르게 샘플링됩니다.
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids=input_ids,
                            max_length=args.max_length,
                            temperature=temp,
                            top_p=0.9,
                            eos_token_id=eos_token_id,
                        )
                    
                    generated_text = tokenizer.decode(output_ids[0].tolist())
                    
                    # 결과 저장용 딕셔너리 (sample_index 추가)
                    result_entry = {
                        "prompt_id": item["prompt_id"],
                        "sample_index": sample_idx + 1,
                        "level": item["level"],
                        "category": item["category"],
                        "temperature": temp,
                        "prompt_text": prompt_text,
                        "generated_text": generated_text,
                        "target_names": item["target_names"],         # <-- 새로 추가
                        "forbidden_names": item["forbidden_names"]    # <-- 새로 추가
                    }
                    results.append(result_entry)
                    pbar.update(1)

    # 결과를 JSON 파일로 예쁘게 저장
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Generation complete! Saved {total_iterations} results to '{args.output_file}'.")

if __name__ == '__main__':
    main()