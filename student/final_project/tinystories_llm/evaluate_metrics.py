import json
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

print("Loading NLP model (spacy)...")
nlp = spacy.load("en_core_web_sm")

def calc_accuracy(text, target_names, forbidden_names):
    """
    Accuracy 계산: 
    1) 타겟 이름이 모두 존재하는가? (Survival)
    2) 금지된 이름이 등장하지 않았는가? (No Intrusion)
    """
    text_lower = text.lower()
    doc = nlp(text)
    # 단어 단위로 분리 (구두점 제외)
    tokens_lower = [token.text.lower() for token in doc if not token.is_punct]
    
    # 1. 생존 검사 (타겟 이름이 하나라도 없으면 0점)
    for t in target_names:
        if t.lower() not in text_lower: # 이름은 서브스트링으로라도 존재해야 함
            return 0
            
    # 2. 침투 검사 (금지된 이름이 명확한 단어로 등장하면 0점)
    for f in forbidden_names:
        if f.lower() in tokens_lower:
            return 0
            
    return 1 # 모든 조건을 만족하면 1점!

def calc_distinct_2(text):
    """
    Diversity 1 (Intra-story): Distinct-2 (고유 Bigram 비율)
    값이 클수록 한 이야기 내에서 다양한 어휘를 사용함.
    """
    tokens = [token.text.lower() for token in nlp(text) if not token.is_punct]
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return len(set(bigrams)) / len(bigrams)

def calc_inter_diversity(texts):
    """
    Diversity 2 (Inter-story): 1 - Mean Pairwise Cosine Similarity
    값이 클수록 같은 프롬프트에서 나온 10개의 이야기가 서로 다름.
    """
    if len(texts) < 2:
        return 0.0
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # 대각선(자기 자신과의 유사도 1)을 제외한 상단 삼각형 값만 추출
        indices = np.triu_indices_from(sim_matrix, k=1)
        mean_sim = np.mean(sim_matrix[indices])
        
        # 직관성을 위해 1에서 빼줌 (높을수록 다양함)
        return 1.0 - mean_sim
    except ValueError:
        # 텍스트가 비어있거나 유효한 단어가 없는 경우
        return 0.0

def main():
    input_file = 'base_model_generated_results.json'
    
    print(f"Reading generated data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    print("Calculating Accuracy and Diversity 1 (Intra-story)...")
    # 행별로 Accuracy와 Distinct-2 계산
    df['accuracy'] = df.apply(lambda row: calc_accuracy(row['generated_text'], row['target_names'], row['forbidden_names']), axis=1)
    df['diversity_1_distinct2'] = df['generated_text'].apply(calc_distinct_2)
    
    print("Calculating Diversity 2 (Inter-story)...")
    # 프롬프트와 온도별로 그룹화하여 10개의 샘플 텍스트를 모은 뒤 계산
    grouped = df.groupby(['prompt_id', 'temperature'])['generated_text'].apply(list).reset_index()
    grouped['diversity_2_inter'] = grouped['generated_text'].apply(calc_inter_diversity)
    
    # 결과를 원본 데이터프레임에 병합
    df = pd.merge(df, grouped[['prompt_id', 'temperature', 'diversity_2_inter']], on=['prompt_id', 'temperature'], how='left')
    
    # --- 분석 결과 요약 ---
    summary_df = df.groupby('temperature').agg({
        'accuracy': 'mean',
        'diversity_1_distinct2': 'mean',
        'diversity_2_inter': 'mean'
    }).reset_index()
    
    print("\n" + "="*50)
    print("📊 Evaluation Summary by Temperature")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50)
    
    # CSV로 상세 결과 저장
    output_csv = 'evaluation_metrics_results.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Detailed results saved to '{output_csv}'")
    
    # --- 프레젠테이션용 그래프 그리기 ---
    print("Generating plots for presentation...")
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Accuracy Plot
    sns.lineplot(data=df, x='temperature', y='accuracy', marker='o', ax=axes[0], color='crimson', linewidth=2.5)
    axes[0].set_title('Accuracy (Coherence)\nTarget Survival & No Intrusion', fontweight='bold')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Accuracy Score')
    axes[0].set_ylim(-0.05, 1.05)
    
    # 2. Diversity 1 Plot
    sns.lineplot(data=df, x='temperature', y='diversity_1_distinct2', marker='s', ax=axes[1], color='dodgerblue', linewidth=2.5)
    axes[1].set_title('Diversity 1 (Intra-story)\nDistinct-2 Bigram Ratio', fontweight='bold')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Distinct-2 Score')
    
    # 3. Diversity 2 Plot
    sns.lineplot(data=summary_df, x='temperature', y='diversity_2_inter', marker='^', ax=axes[2], color='forestgreen', linewidth=2.5)
    axes[2].set_title('Diversity 2 (Inter-story)\n1 - Cosine Similarity', fontweight='bold')
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Diversity Score (Higher = More Unique)')
    
    plt.tight_layout()
    plot_file = 'temperature_vs_metrics_plot.png'
    plt.savefig(plot_file, dpi=300)
    print(f"✅ Plot saved to '{plot_file}'")

if __name__ == '__main__':
    main()