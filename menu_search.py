#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
순수 AI 검색 엔진 v3.0 - LLM 의존성 제거
- Bi-encoder (multilingual-e5-small) + BM25 하이브리드 검색
- Cross-encoder (ms-marco-MiniLM-L-12-v2) Rerank
- 2단계 앙상블 검색 시스템
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import time
import re
from collections import Counter
import math
import ssl

warnings.filterwarnings("ignore")

# SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context


class BM25:
    """BM25 알고리즘 구현 (키워드 검색용)"""
    
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.avgdl = 0
        
        self._initialize()
    
    def _initialize(self):
        """BM25 초기화"""
        for doc in self.corpus:
            words = doc.lower().split()
            self.doc_lens.append(len(words))
            
            word_freq = Counter(words)
            self.doc_freqs.append(word_freq)
        
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        
        all_words = set()
        for doc_freq in self.doc_freqs:
            all_words.update(doc_freq.keys())
        
        for word in all_words:
            containing_docs = sum(1 for doc_freq in self.doc_freqs if word in doc_freq)
            self.idf[word] = math.log((len(self.corpus) - containing_docs + 0.5) / (containing_docs + 0.5))
    
    def get_scores(self, query):
        """쿼리에 대한 모든 문서의 BM25 점수 계산"""
        query_words = query.lower().split()
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_lens[i]
            
            for word in query_words:
                if word in doc_freq:
                    freq = doc_freq[word]
                    idf = self.idf.get(word, 0)
                    
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores


class PureSearchEngine:
    """순수 검색 엔진 - LLM 의존성 없음"""
    
    def __init__(self):
        """검색 엔진 초기화"""
        print("🚀 순수 AI 검색 엔진 v3.0 초기화")
        print("="*50)
        
        # 데이터 로드
        self.data = self._load_data()
        
        # 필수 모델 로드 (실패시 종료)
        self._load_required_models()
        
        # 검색 인덱스 생성
        self._create_search_indices()
        
        print("✅ 검색 엔진 초기화 완료!")
    
    def _load_data(self):
        """데이터 로드"""
        try:
            with open('ia-data_enhanced.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # PRIMARY, SECONDARY 페이지만 유지
            filtered_data = [item for item in data if item.get('page_classification') in ['PRIMARY', 'SECONDARY']]
            
            print(f"📊 데이터 로드: {len(filtered_data)}개 항목")
            return filtered_data
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            exit(1)
    
    def _load_required_models(self):
        """필수 모델 로드 - 실패시 프로그램 종료"""
        
        # 1. Bi-encoder 모델 로드
        bi_encoder_path = "./models/multilingual-e5-small"
        if not os.path.exists(bi_encoder_path):
            print(f"❌ Bi-encoder 모델을 찾을 수 없습니다: {bi_encoder_path}")
            exit(1)
        
        try:
            print("🔄 Bi-encoder 모델 로드 중...")
            self.bi_encoder = SentenceTransformer(bi_encoder_path)
            print("✅ multilingual-e5-small 로드 완료")
        except Exception as e:
            print(f"❌ Bi-encoder 로드 실패: {e}")
            exit(1)
        
        # 2. Cross-encoder 모델 로드
        cross_encoder_path = "./models/ms-marco-MiniLM-L-12-v2"
        if not os.path.exists(cross_encoder_path):
            print(f"❌ Cross-encoder 모델을 찾을 수 없습니다: {cross_encoder_path}")
            exit(1)
        
        try:
            print("🔄 Cross-encoder 모델 로드 중...")
            self.cross_encoder = CrossEncoder(cross_encoder_path)
            print("✅ ms-marco-MiniLM-L-12-v2 로드 완료")
        except Exception as e:
            print(f"❌ Cross-encoder 로드 실패: {e}")
            exit(1)
    
    def _create_search_indices(self):
        """검색 인덱스 생성"""
        print("🔍 검색 인덱스 생성 중...")
        
        # 1. Bi-encoder 임베딩 생성
        texts = []
        for item in self.data:
            # 강화된 텍스트 생성
            parts = [
                item.get('page_name', ''),
                item.get('Category', ''),
                item.get('Service', ''),
                ' '.join(item.get('hierarchy', [])),
                ' '.join(item.get('representative_keywords', [])),
                item.get('ai_description', '')
            ]
            text = ' '.join(filter(None, parts))
            texts.append(text)
        
        self.embeddings = self.bi_encoder.encode(texts, show_progress_bar=True)
        print(f"✅ {len(self.embeddings)}개 임베딩 생성 완료")
        
        # 2. BM25 인덱스 생성
        bm25_texts = []
        for item in self.data:
            parts = [
                item.get('page_name', ''),
                item.get('Category', ''),
                item.get('Service', ''),
                ' '.join(item.get('representative_keywords', []))
            ]
            text = ' '.join(filter(None, parts))
            bm25_texts.append(text)
        
        self.bm25 = BM25(bm25_texts)
        print("✅ BM25 인덱스 생성 완료")
    
    def hybrid_search(self, query, top_k=10, vector_weight=0.7, keyword_weight=0.3):
        """1단계: 하이브리드 검색 (Bi-encoder + BM25)"""
        
        # 벡터 검색
        query_embedding = self.bi_encoder.encode([query])
        vector_similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 키워드 검색
        keyword_scores = self.bm25.get_scores(query)
        
        # 점수 정규화
        max_vector = max(vector_similarities) if max(vector_similarities) > 0 else 1
        max_keyword = max(keyword_scores) if max(keyword_scores) > 0 else 1
        
        # 하이브리드 점수 계산
        results = []
        for i, (vector_sim, keyword_score) in enumerate(zip(vector_similarities, keyword_scores)):
            normalized_vector = vector_sim / max_vector
            normalized_keyword = keyword_score / max_keyword
            
            # PRIMARY 페이지 가중치
            item = self.data[i]
            classification_boost = 1.2 if item.get('page_classification') == 'PRIMARY' else 1.0
            
            hybrid_score = ((normalized_vector * vector_weight) + 
                          (normalized_keyword * keyword_weight)) * classification_boost
            
            results.append({
                'index': i,
                'data': item,
                'vector_score': float(vector_sim),
                'keyword_score': float(keyword_score),
                'hybrid_score': float(hybrid_score),
                'classification_boost': classification_boost
            })
        
        # 점수순 정렬
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:top_k]
    
    def rerank_candidates(self, query, candidates, top_k=5):
        """2단계: Cross-encoder Rerank"""
        if not candidates:
            return []
        
        print(f"�� Cross-encoder Rerank: {len(candidates)}개 후보 재순위화")
        
        # 쿼리-문서 쌍 생성
        query_doc_pairs = []
        for candidate in candidates:
            item = candidate['data']
            # 간결한 문서 텍스트 생성
            doc_text = f"{item.get('page_name', '')} {item.get('ai_description', '')}"
            query_doc_pairs.append([query, doc_text])
        
        # Cross-encoder 점수 계산
        rerank_scores = self.cross_encoder.predict(query_doc_pairs)
        
        # 점수 정규화 및 결합
        min_rerank = min(rerank_scores)
        max_rerank = max(rerank_scores)
        score_range = max_rerank - min_rerank if max_rerank > min_rerank else 1
        
        for i, candidate in enumerate(candidates):
            raw_rerank = float(rerank_scores[i])
            normalized_rerank = (raw_rerank - min_rerank) / score_range
            hybrid_score = candidate['hybrid_score']
            
            # 최종 점수: Rerank 70% + Hybrid 30%
            final_score = (normalized_rerank * 0.7) + (hybrid_score * 0.3)
            
            candidate['rerank_score'] = raw_rerank
            candidate['final_score'] = final_score
        
        # 최종 점수로 재정렬
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        
        print(f"   상위 {min(top_k, len(reranked))}개 결과:")
        for i, result in enumerate(reranked[:top_k], 1):
            item = result['data']
            print(f"   {i}. {item.get('page_name', '')} (최종: {result['final_score']:.3f})")
        
        return reranked[:top_k]
    
    def search(self, query):
        """통합 검색 실행"""
        print(f"\n{'='*60}")
        print(f"🔍 검색 쿼리: '{query}'")
        print(f"{'='*60}")
        
        # 1단계: 하이브리드 검색
        print("\n🔍 1단계: 하이브리드 검색 (Bi-encoder + BM25)")
        candidates = self.hybrid_search(query, top_k=10)
        
        if not candidates:
            print("❌ 검색 결과가 없습니다.")
            return None
        
        print(f"   후보 {len(candidates)}개 발견")
        for i, result in enumerate(candidates[:5], 1):  # 상위 5개만 출력
            item = result['data']
            print(f"   {i}. {item.get('page_name', '')} (하이브리드: {result['hybrid_score']:.3f})")
        
        # 2단계: Cross-encoder Rerank
        print(f"\n🎯 2단계: Cross-encoder Rerank")
        final_results = self.rerank_candidates(query, candidates, top_k=5)
        
        return final_results[0] if final_results else None
    
    def print_result(self, query, result):
        """검색 결과 출력"""
        if not result:
            print("❌ 검색 결과가 없습니다.")
            return
        
        item = result['data']
        print(f"\n{'='*60}")
        print(f"🏆 최종 검색 결과")
        print(f"{'='*60}")
        print(f"📍 페이지명: {item.get('page_name', '')}")
        print(f"📂 카테고리: {item.get('Category', '')}")
        print(f"🏢 서비스: {item.get('Service', '')}")
        
        if item.get('hierarchy'):
            print(f"📋 계층구조: {' > '.join(item.get('hierarchy', []))}")
        
        # 점수 정보
        print(f"📊 최종 점수: {result.get('final_score', 0):.4f}")
        print(f"   ┣ Rerank 점수: {result.get('rerank_score', 0):.4f}")
        print(f"   ┗ 하이브리드 점수: {result.get('hybrid_score', 0):.4f}")
        
        if result.get('classification_boost', 1.0) > 1.0:
            print(f"   ⭐ PRIMARY 가중치 적용")
        
        if item.get('ai_description'):
            print(f"📝 설명: {item.get('ai_description', '')}")
        
        if item.get('representative_keywords'):
            print(f"🔍 관련 키워드: {', '.join(item.get('representative_keywords', []))}")
        
        print(f"{'='*60}")
    
    def get_statistics(self):
        """시스템 통계"""
        print(f"\n{'='*60}")
        print(f"📊 시스템 통계")
        print(f"{'='*60}")
        print(f"📄 총 데이터: {len(self.data)}개 항목")
        print(f"🧠 Bi-encoder: multilingual-e5-small")
        print(f"🎯 Cross-encoder: ms-marco-MiniLM-L-12-v2")
        
        # 분류 통계
        primary_count = sum(1 for item in self.data if item.get('page_classification') == 'PRIMARY')
        secondary_count = sum(1 for item in self.data if item.get('page_classification') == 'SECONDARY')
        
        print(f"🎯 PRIMARY 페이지: {primary_count}개 (가중치 x1.2)")
        print(f"📋 SECONDARY 페이지: {secondary_count}개")
        print(f"⚖️ 하이브리드 가중치: 벡터 70% + 키워드 30%")
        print(f"🔄 Rerank 가중치: Rerank 70% + 하이브리드 30%")
        print(f"{'='*60}")


def main():
    """메인 함수"""
    try:
        # 검색 엔진 초기화
        engine = PureSearchEngine()
        
        # 통계 출력
        engine.get_statistics()
        
        print("\n자연어로 검색어를 입력하세요 ('q' 입력시 종료, 'stats' 입력시 통계 조회):")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if query.lower() == 'q':
                    print("프로그램을 종료합니다.")
                    break
                
                if query.lower() == 'stats':
                    engine.get_statistics()
                    continue
                
                if not query:
                    print("검색어를 입력해주세요.")
                    continue
                
                # 검색 수행
                start_time = time.time()
                result = engine.search(query)
                search_time = time.time() - start_time
                
                engine.print_result(query, result)
                print(f"⏱️ 검색 시간: {search_time:.2f}초")
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류가 발생했습니다: {e}")
    
    except Exception as e:
        print(f"❌ 초기화 중 오류가 발생했습니다: {e}")
        exit(1)


if __name__ == "__main__":
    main()
