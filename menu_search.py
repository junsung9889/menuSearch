#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메뉴 검색 프로그램
벡터 임베딩을 통한 유사도 검색
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
warnings.filterwarnings("ignore")

# SSL 인증서 문제 해결
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class MenuSearcher:
    def __init__(self, json_file_path):
        """
        메뉴 검색기 초기화
        """
        # 다운로드된 로컬 모델 경로들 확인 (성능 순서대로)
        local_model_paths = [
            "./models/multilingual-e5-small",  # 🥇 최고 성능 (평균 유사도: 0.9448)
            "./models/ko-sroberta-multitask",  # 🥈 한국어 특화 (평균 유사도: 0.8523)
            "./models/distiluse-base-multilingual-cased-v2",  # 🥉 빠른 속도
            "./models/all-mpnet-base-v2",  # 영어 중심 고성능
            "./manual_model",  # 기존 다운로드된 모델
            "./models/paraphrase-multilingual-MiniLM-L12-v2",  # 다국어 BERT
            "./local_model",   # HuggingFace Hub으로 다운로드된 모델
            "./paraphrase-multilingual-MiniLM-L12-v2"  # Git으로 clone된 모델
        ]
        
        model_loaded = False
        self.current_model_name = "Unknown"
        
        # 로컬 모델들 순서대로 시도
        for model_path in local_model_paths:
            if os.path.exists(model_path) and os.path.exists(f"{model_path}/pytorch_model.bin"):
                try:
                    model_name = os.path.basename(model_path)
                    print(f"🔄 {model_name} 모델을 로드하고 있습니다...")
                    self.model = SentenceTransformer(model_path)
                    print(f"✅ {model_name} 모델 로드 완료!")
                    
                    # 모델별 성능 정보 표시
                    if "multilingual-e5-small" in model_path:
                        print("   🏆 최고 성능 모델 (평균 유사도: 94.48%)")
                    elif "ko-sroberta" in model_path:
                        print("   🇰🇷 한국어 특화 모델 (평균 유사도: 85.23%)")
                    elif "distiluse" in model_path:
                        print("   ⚡ 고속 다국어 모델")
                    else:
                        print("   📦 다국어 지원 모델")
                    
                    self.use_tfidf = False
                    self.current_model_name = model_name
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"❌ {model_path} 로드 실패: {e}")
                    continue
        
        # 로컬 모델이 없으면 온라인 시도
        if not model_loaded:
            try:
                print("온라인 모델을 로드하고 있습니다...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("모델 로드 완료")
                self.use_tfidf = False
                self.current_model_name = "all-MiniLM-L6-v2"
            except Exception as e:
                print(f"온라인 모델 로드 중 오류 발생: {e}")
                print("대체 방법으로 TF-IDF를 사용합니다.")
                self.use_tfidf = True
                self.current_model_name = "TF-IDF"
            
        self.data = self.load_data(json_file_path)
        if not self.use_tfidf:
            self.embeddings = self.create_embeddings()
        else:
            self.setup_tfidf()
    
    def load_data(self, json_file_path):
        """
        JSON 데이터 로드
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"총 {len(data)}개의 항목을 로드했습니다.")
            return data
        except Exception as e:
            print(f"데이터 로딩 중 오류가 발생했습니다: {e}")
            return []
    
    def setup_tfidf(self):
        """
        TF-IDF 설정 (백업 방법)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [self.create_text_for_embedding(item) for item in self.data]
        page_texts = [item.get('page_name', '') for item in self.data]
        full_context_texts = [f"{item.get('Category', '')} {item.get('Service', '')} {' '.join(item.get('hierarchy', []))}" for item in self.data]
        
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_vectorizer_page = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_vectorizer_context = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.tfidf_matrix_page = self.tfidf_vectorizer_page.fit_transform(page_texts)
        self.tfidf_matrix_context = self.tfidf_vectorizer_context.fit_transform(full_context_texts)
    
    def create_text_for_embedding(self, item):
        """
        각 항목에 대해 임베딩용 텍스트 생성
        """
        category = item.get('Category', '')
        service = item.get('Service', '')
        page_name = item.get('page_name', '')
        hierarchy = ' > '.join(item.get('hierarchy', []))
        
        # 전체 텍스트 조합
        full_text = f"카테고리: {category} 서비스: {service} 페이지명: {page_name} 계층구조: {hierarchy}"
        return full_text.strip()
    
    def create_embeddings(self):
        """
        모든 항목에 대해 벡터 임베딩 생성
        """
        print("벡터 임베딩을 생성하고 있습니다...")
        texts = [self.create_text_for_embedding(item) for item in self.data]
        
        # 각 필드별로도 개별 임베딩 생성
        page_texts = [item.get('page_name', '') for item in self.data]
        full_context_texts = [f"{item.get('Category', '')} {item.get('Service', '')} {' '.join(item.get('hierarchy', []))}" for item in self.data]
        
        embeddings = {
            'full': self.model.encode(texts, show_progress_bar=True),
            'page': self.model.encode(page_texts, show_progress_bar=True),
            'context': self.model.encode(full_context_texts, show_progress_bar=True)
        }
        
        print("✅ 벡터 임베딩 생성이 완료되었습니다.")
        return embeddings
    
    def search(self, query, top_k=3):
        """
        검색어에 대해 유사도 검색 수행
        """
        if not self.data:
            return []
        
        if self.use_tfidf:
            return self.search_with_tfidf(query, top_k)
        else:
            return self.search_with_embeddings(query, top_k)
    
    def search_with_embeddings(self, query, top_k):
        """
        벡터 임베딩을 사용한 검색
        """
        # 검색어 임베딩
        query_embedding_full = self.model.encode([query])
        query_embedding_page = self.model.encode([query])
        query_embedding_context = self.model.encode([query])
        
        # 각 유형별 유사도 계산
        similarities_full = cosine_similarity(query_embedding_full, self.embeddings['full'])[0]
        similarities_page = cosine_similarity(query_embedding_page, self.embeddings['page'])[0]
        similarities_context = cosine_similarity(query_embedding_context, self.embeddings['context'])[0]
        
        # 결과 조합 (가중치 적용)
        results = []
        for i, (sim_full, sim_page, sim_context) in enumerate(zip(similarities_full, similarities_page, similarities_context)):
            # 가중 평균으로 종합 점수 계산
            combined_score = (sim_full * 0.5) + (sim_page * 0.3) + (sim_context * 0.2)
            
            results.append({
                'index': i,
                'data': self.data[i],
                'similarity_full': float(sim_full),
                'similarity_page': float(sim_page),
                'similarity_context': float(sim_context),
                'combined_score': float(combined_score)
            })
        
        # 종합 점수 기준으로 정렬하여 상위 k개 반환
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]
    
    def search_with_tfidf(self, query, top_k):
        """
        TF-IDF를 사용한 검색 (백업 방법)
        """
        # 검색어를 TF-IDF 벡터로 변환
        query_vector = self.tfidf_vectorizer.transform([query])
        query_vector_page = self.tfidf_vectorizer_page.transform([query])
        query_vector_context = self.tfidf_vectorizer_context.transform([query])
        
        # 유사도 계산
        similarities_full = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        similarities_page = cosine_similarity(query_vector_page, self.tfidf_matrix_page)[0]
        similarities_context = cosine_similarity(query_vector_context, self.tfidf_matrix_context)[0]
        
        # 결과 조합
        results = []
        for i, (sim_full, sim_page, sim_context) in enumerate(zip(similarities_full, similarities_page, similarities_context)):
            results.append({
                'index': i,
                'data': self.data[i],
                'similarity_full': float(sim_full),
                'similarity_page': float(sim_page),
                'similarity_context': float(sim_context),
                'combined_score': float(sim_full)  # 전체 유사도를 기준으로 정렬
            })
        
        # 전체 유사도 기준으로 정렬하여 상위 k개 반환
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]
    
    def print_results(self, query, results):
        """
        검색 결과 출력
        """
        print(f"\n검색어: {query}")
        print(f"사용 모델: {self.current_model_name}")
        print(f"\n유사한 항목들:")
        
        for i, result in enumerate(results, 1):
            item = result['data']
            print(f"\n{i}. 전체 유사도: {result['similarity_full']:.4f}")
            print(f"   - 페이지별 유사도: {result['similarity_page']:.4f}")
            print(f"   - 전체 컨텍스트 유사도: {result['similarity_context']:.4f}")
            if not self.use_tfidf:
                print(f"   - 종합 점수: {result['combined_score']:.4f}")
            print(f"   카테고리: {item.get('Category', '')}")
            print(f"   서비스: {item.get('Service', '')}")
            print(f"   페이지명: {item.get('page_name', '')}")


def main():
    """
    메인 함수
    """
    searcher = MenuSearcher('ia-data.json')
    
    print("검색어를 입력하세요 ('q' 입력시 종료):")
    
    while True:
        try:
            query = input("> ").strip()
            
            if query.lower() == 'q':
                print("프로그램을 종료합니다.")
                break
            
            if not query:
                print("검색어를 입력해주세요.")
                continue
            
            # 검색 수행
            results = searcher.search(query, top_k=3)
            
            if results:
                searcher.print_results(query, results)
            else:
                print("검색 결과가 없습니다.")
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main() 