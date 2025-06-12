#!/usr/bin/env python3
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
        print(f"\n🔍 {model_name} 테스트 중...")
        
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
    print("\n" + "=" * 80)
    print("📈 모델 성능 비교 결과")
    print("=" * 80)
    
    # 정렬 (평균 유사도 기준)
    results.sort(key=lambda x: x["avg_similarity"], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"\n🏆 {i}위: {result['model_name']}")
        print(f"   ⚡ 로드 시간: {result['load_time']:.2f}초")
        print(f"   🔄 임베딩 시간: {result['embed_time']:.3f}초")
        print(f"   📊 평균 유사도: {result['avg_similarity']:.4f}")
        print("   🔍 쿼리별 결과:")
        for query, sim in result['results'].items():
            print(f"      • {query}: {sim:.4f}")

if __name__ == "__main__":
    compare_models()
