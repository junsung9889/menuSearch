#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
여러 모델 다운로드 및 비교 스크립트
"""

import os
import shutil
import urllib.request
import ssl
from pathlib import Path

# SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context

# 한국어 지원이 좋은 모델들
MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "다국어 지원, 경량화된 BERT 기반 모델",
        "size": "약 470MB"
    },
    "distiluse-base-multilingual-cased-v2": {
        "repo": "sentence-transformers/distiluse-base-multilingual-cased-v2", 
        "description": "다국어 DistilBERT, 빠른 속도",
        "size": "약 540MB"
    },
    "all-mpnet-base-v2": {
        "repo": "sentence-transformers/all-mpnet-base-v2",
        "description": "MPNet 기반, 높은 성능 (영어 중심)",
        "size": "약 420MB"
    },
    "ko-sroberta-multitask": {
        "repo": "jhgan/ko-sroberta-multitask",
        "description": "한국어 특화 RoBERTa",
        "size": "약 450MB"
    },
    "multilingual-e5-small": {
        "repo": "intfloat/multilingual-e5-small", 
        "description": "최신 E5 다국어 모델, 작은 크기",
        "size": "약 470MB"
    }
}

def download_model(model_name, repo_id):
    """
    개별 모델 다운로드
    """
    try:
        print(f"\n🔄 {model_name} 모델 다운로드 중...")
        print(f"   📦 {MODELS[model_name]['description']}")
        print(f"   💾 크기: {MODELS[model_name]['size']}")
        
        model_dir = f"models/{model_name}"
        
        # 기존 폴더가 있으면 건너뛰기
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/pytorch_model.bin"):
            print(f"   ✅ 이미 다운로드됨: {model_dir}")
            return model_dir
        
        # 폴더 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 필수 파일들
        base_url = f"https://huggingface.co/{repo_id}/resolve/main"
        
        files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
        
        # sentence-transformers 전용 파일들
        if "sentence-transformers/" in repo_id:
            files.extend([
                "sentence_bert_config.json",
                "modules.json"
            ])
        
        success_count = 0
        for filename in files:
            try:
                url = f"{base_url}/{filename}"
                filepath = f"{model_dir}/{filename}"
                
                urllib.request.urlretrieve(url, filepath)
                success_count += 1
                
            except Exception as e:
                print(f"   ⚠️  {filename} 다운로드 실패: {e}")
                continue
        
        if success_count >= 3:  # 최소 3개 파일이 성공하면 OK
            print(f"   ✅ 다운로드 완료: {model_dir} ({success_count}/{len(files)} 파일)")
            return model_dir
        else:
            print(f"   ❌ 다운로드 실패: 필수 파일 부족")
            return None
            
    except Exception as e:
        print(f"   ❌ {model_name} 다운로드 실패: {e}")
        return None

def download_all_models():
    """
    모든 모델 다운로드
    """
    print("🚀 여러 모델 다운로드를 시작합니다...")
    print("=" * 60)
    
    downloaded_models = []
    
    for model_name, info in MODELS.items():
        model_path = download_model(model_name, info["repo"])
        if model_path:
            downloaded_models.append((model_name, model_path))
    
    print("\n" + "=" * 60)
    print("📊 다운로드 결과:")
    print(f"✅ 성공: {len(downloaded_models)}개 모델")
    print(f"❌ 실패: {len(MODELS) - len(downloaded_models)}개 모델")
    
    if downloaded_models:
        print("\n📂 다운로드된 모델들:")
        for model_name, model_path in downloaded_models:
            print(f"   • {model_name}: {model_path}")
    
    return downloaded_models

def create_model_comparison_script():
    """
    모델 비교 스크립트 생성
    """
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 성능 비교 스크립트
"""

import json
import os
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data():
    """JSON 데이터 로드"""
    with open('ia-data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def test_model(model_path, model_name, test_queries):
    """개별 모델 테스트"""
    try:
        print(f"\\n🔍 {model_name} 테스트 중...")
        
        # 모델 로드
        start_time = time.time()
        model = SentenceTransformer(model_path)
        load_time = time.time() - start_time
        
        # 샘플 데이터로 테스트
        sample_texts = [
            "카드 이용내역 조회",
            "포인트 적립 확인", 
            "결제 내역 조회",
            "카드 발급 신청",
            "로그인 인증"
        ]
        
        # 임베딩 생성 시간 측정
        start_time = time.time()
        embeddings = model.encode(sample_texts)
        embed_time = time.time() - start_time
        
        # 테스트 쿼리들에 대한 검색 성능
        results = {}
        for query in test_queries:
            query_emb = model.encode([query])
            similarities = cosine_similarity(query_emb, embeddings)[0]
            max_sim = float(np.max(similarities))
            results[query] = max_sim
        
        return {
            "model_name": model_name,
            "load_time": load_time,
            "embed_time": embed_time,
            "avg_similarity": np.mean(list(results.values())),
            "results": results
        }
        
    except Exception as e:
        print(f"   ❌ {model_name} 테스트 실패: {e}")
        return None

def compare_models():
    """모든 모델 성능 비교"""
    print("🏁 모델 성능 비교를 시작합니다...")
    
    # 테스트 쿼리들
    test_queries = [
        "카드 이용내역",
        "포인트 조회", 
        "결제 확인",
        "로그인",
        "회원가입"
    ]
    
    # 다운로드된 모델들 찾기
    model_dirs = []
    if os.path.exists("models"):
        for item in os.listdir("models"):
            model_path = f"models/{item}"
            if os.path.isdir(model_path) and os.path.exists(f"{model_path}/pytorch_model.bin"):
                model_dirs.append((item, model_path))
    
    # 기존 manual_model도 포함
    if os.path.exists("manual_model/pytorch_model.bin"):
        model_dirs.append(("manual_model", "manual_model"))
    
    if not model_dirs:
        print("❌ 테스트할 모델이 없습니다.")
        return
    
    print(f"📊 총 {len(model_dirs)}개 모델을 테스트합니다...")
    
    results = []
    for model_name, model_path in model_dirs:
        result = test_model(model_path, model_name, test_queries)
        if result:
            results.append(result)
    
    # 결과 출력
    print("\\n" + "=" * 80)
    print("📈 모델 성능 비교 결과")
    print("=" * 80)
    
    # 정렬 (평균 유사도 기준)
    results.sort(key=lambda x: x["avg_similarity"], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"\\n🏆 {i}위: {result['model_name']}")
        print(f"   ⚡ 로드 시간: {result['load_time']:.2f}초")
        print(f"   🔄 임베딩 시간: {result['embed_time']:.3f}초")
        print(f"   📊 평균 유사도: {result['avg_similarity']:.4f}")
        print("   🔍 쿼리별 결과:")
        for query, sim in result['results'].items():
            print(f"      • {query}: {sim:.4f}")

if __name__ == "__main__":
    compare_models()
'''
    
    with open("compare_models.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 모델 비교 스크립트 생성 완료: compare_models.py")

def main():
    print("🎯 다중 모델 다운로드 도구")
    print("=" * 60)
    
    print("📋 다운로드할 모델 목록:")
    for i, (name, info) in enumerate(MODELS.items(), 1):
        print(f"{i}. {name}")
        print(f"   📝 {info['description']}")
        print(f"   💾 {info['size']}")
        print()
    
    response = input("모든 모델을 다운로드하시겠습니까? (y/n): ").lower()
    
    if response == 'y':
        downloaded_models = download_all_models()
        
        if downloaded_models:
            create_model_comparison_script()
            print("\n🎉 완료!")
            print("💡 다음 명령어로 모델들을 비교해보세요:")
            print("   python3 compare_models.py")
    else:
        print("❌ 다운로드를 취소했습니다.")

if __name__ == "__main__":
    main() 