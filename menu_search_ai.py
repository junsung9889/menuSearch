#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API를 활용한 지능형 메뉴 검색 프로그램
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from openai import OpenAI
import time
warnings.filterwarnings("ignore")

# SSL 인증서 문제 해결
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class AIMenuSearcher:
    def __init__(self, json_file_path, openai_api_key=None):
        """
        AI 메뉴 검색기 초기화
        """
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        
        # 다운로드된 로컬 모델 경로들 확인 (성능 순서대로)
        local_model_paths = [
            "./models/multilingual-e5-small",
            "./models/ko-sroberta-multitask",
            "./models/distiluse-base-multilingual-cased-v2",
            "./models/all-mpnet-base-v2",
            "./manual_model",
            "./models/paraphrase-multilingual-MiniLM-L12-v2",
            "./local_model",
            "./paraphrase-multilingual-MiniLM-L12-v2"
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
        
        # 대표 검색어 생성 및 로드
        self.enhanced_data_path = json_file_path.replace('.json', '_enhanced.json')
        self.data = self.enhance_data_with_keywords()
        
        if not self.use_tfidf:
            self.embeddings = self.create_embeddings()
        else:
            self.setup_tfidf()
    
    def extract_keyword_with_ai(self, query):
        """
        OpenAI API를 사용해 자연어 입력을 가장 잘 요약하는 한 단어 생성
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 한국어 자연어 입력을 가장 잘 요약하는 전문가입니다. 사용자의 질문이나 요청을 전체적으로 가장 잘 표현하는 한 단어를 만들어주세요. 필요하다면 여러 단어를 조합해서 복합어로 만들어도 됩니다. 예: '카드'+'이용내역'='카드이용내역'. 반드시 한 단어로만 답하세요."},
                    {"role": "user", "content": f"다음 문장을 가장 잘 요약하는 한 단어를 만들어주세요: '{query}'"}
                ],
                max_tokens=20,
                temperature=0.1
            )
            keyword = response.choices[0].message.content.strip()
            print(f"🔍 AI 요약 키워드: '{query}' → '{keyword}'")
            return keyword
        except Exception as e:
            print(f"❌ AI 요약 키워드 생성 실패: {e}")
            return query  # 실패시 원본 쿼리 반환
    
    def select_best_menu_with_ai(self, original_query, menu_candidates):
        """
        OpenAI API를 사용해 최종 메뉴 하나 선택
        """
        try:
            # 메뉴 후보들을 텍스트로 포맷팅
            menu_text = ""
            for i, result in enumerate(menu_candidates, 1):
                item = result['data']
                menu_text += f"{i}. {item.get('page_name', '')} (카테고리: {item.get('Category', '')}, 서비스: {item.get('Service', '')})\n"
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 사용자의 의도를 파악해 가장 적합한 메뉴를 선택하는 전문가입니다. 사용자의 원래 질문과 메뉴 후보들을 보고, 가장 관련성이 높은 메뉴 하나의 번호만 답하세요. 반드시 숫자만 답하세요."},
                    {"role": "user", "content": f"사용자 질문: '{original_query}'\n\n메뉴 후보들:\n{menu_text}\n\n가장 적합한 메뉴의 번호를 선택하세요:"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            selected_num = int(response.choices[0].message.content.strip())
            if 1 <= selected_num <= len(menu_candidates):
                selected_menu = menu_candidates[selected_num - 1]
                print(f"🎯 AI 최종 선택: {selected_num}번 - {selected_menu['data'].get('page_name', '')}")
                return selected_menu
            else:
                print(f"⚠️ AI 선택 번호가 범위를 벗어남: {selected_num}")
                return menu_candidates[0]  # 첫 번째 후보 반환
                
        except Exception as e:
            print(f"❌ AI 메뉴 선택 실패: {e}")
            return menu_candidates[0]  # 실패시 첫 번째 후보 반환
    
    def load_data(self, json_file_path):
        """JSON 데이터 로드"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"총 {len(data)}개의 항목을 로드했습니다.")
            return data
        except Exception as e:
            print(f"데이터 로딩 중 오류가 발생했습니다: {e}")
            return []
    
    def setup_tfidf(self):
        """TF-IDF 설정 (백업 방법)"""
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
        """각 항목에 대해 임베딩용 텍스트 생성 (대표 검색어 가중치 포함)"""
        category = item.get('Category', '')
        service = item.get('Service', '')
        page_name = item.get('page_name', '')
        hierarchy = ' > '.join(item.get('hierarchy', []))
        
        # 기본 텍스트
        basic_text = f"카테고리: {category} 서비스: {service} 페이지명: {page_name} 계층구조: {hierarchy}"
        
        # 대표 검색어가 있으면 가중치로 추가 (3번 반복으로 가중치 부여)
        representative_keywords = item.get('representative_keywords', [])
        if representative_keywords:
            keywords_text = ' '.join(representative_keywords)
            # 대표 검색어를 3번 반복하여 가중치 부여
            weighted_keywords = f" {keywords_text} {keywords_text} {keywords_text}"
            full_text = basic_text + weighted_keywords
        else:
            full_text = basic_text
        
        return full_text.strip()
    
    def create_embeddings(self):
        """모든 항목에 대해 벡터 임베딩 생성"""
        print("벡터 임베딩을 생성하고 있습니다...")
        texts = [self.create_text_for_embedding(item) for item in self.data]
        
        page_texts = [item.get('page_name', '') for item in self.data]
        full_context_texts = [f"{item.get('Category', '')} {item.get('Service', '')} {' '.join(item.get('hierarchy', []))}" for item in self.data]
        
        embeddings = {
            'full': self.model.encode(texts, show_progress_bar=True),
            'page': self.model.encode(page_texts, show_progress_bar=True),
            'context': self.model.encode(full_context_texts, show_progress_bar=True)
        }
        
        print("✅ 벡터 임베딩 생성이 완료되었습니다.")
        return embeddings
    
    def search_with_embeddings(self, query, top_k):
        """벡터 임베딩을 사용한 검색"""
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
            combined_score = (sim_full * 0.5) + (sim_page * 0.3) + (sim_context * 0.2)
            
            results.append({
                'index': i,
                'data': self.data[i],
                'similarity_full': float(sim_full),
                'similarity_page': float(sim_page),
                'similarity_context': float(sim_context),
                'combined_score': float(combined_score)
            })
        
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]
    
    def search_with_tfidf(self, query, top_k):
        """TF-IDF를 사용한 검색 (백업 방법)"""
        query_vector = self.tfidf_vectorizer.transform([query])
        query_vector_page = self.tfidf_vectorizer_page.transform([query])
        query_vector_context = self.tfidf_vectorizer_context.transform([query])
        
        similarities_full = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        similarities_page = cosine_similarity(query_vector_page, self.tfidf_matrix_page)[0]
        similarities_context = cosine_similarity(query_vector_context, self.tfidf_matrix_context)[0]
        
        results = []
        for i, (sim_full, sim_page, sim_context) in enumerate(zip(similarities_full, similarities_page, similarities_context)):
            results.append({
                'index': i,
                'data': self.data[i],
                'similarity_full': float(sim_full),
                'similarity_page': float(sim_page),
                'similarity_context': float(sim_context),
                'combined_score': float(sim_full)
            })
        
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]
    
    def ai_search(self, query):
        """
        AI 강화 검색 프로세스
        1. AI로 키워드 추출
        2. 벡터 검색으로 상위 5개 후보 추출
        3. AI로 최종 메뉴 하나 선택
        """
        print(f"\n{'='*60}")
        print(f"🚀 AI 검색 시작: '{query}'")
        print(f"{'='*60}")
        
        # 1단계: AI로 키워드 추출
        print(f"\n📝 1단계: 요약 키워드 생성")
        keyword = self.extract_keyword_with_ai(query)
        
        # 2단계: 벡터 검색으로 상위 5개 후보 추출
        print(f"\n🔍 2단계: 벡터 검색 (상위 5개 후보)")
        if self.use_tfidf:
            candidates = self.search_with_tfidf(keyword, top_k=5)
        else:
            candidates = self.search_with_embeddings(keyword, top_k=5)
        
        if not candidates:
            print("❌ 검색 결과가 없습니다.")
            return None
        
        print(f"   찾은 후보 {len(candidates)}개:")
        for i, result in enumerate(candidates, 1):
            item = result['data']
            print(f"   {i}. {item.get('page_name', '')} (유사도: {result['combined_score']:.4f})")
        
        # 3단계: AI로 최종 메뉴 선택
        print(f"\n🎯 3단계: AI 최종 선택")
        final_result = self.select_best_menu_with_ai(query, candidates)
        
        return final_result
    
    def print_final_result(self, query, result):
        """최종 결과 출력"""
        if not result:
            print("❌ 검색 결과가 없습니다.")
            return
        
        item = result['data']
        print(f"\n{'='*60}")
        print(f"🏆 최종 추천 메뉴")
        print(f"{'='*60}")
        print(f"📍 페이지명: {item.get('page_name', '')}")
        print(f"📂 카테고리: {item.get('Category', '')}")
        print(f"🏢 서비스: {item.get('Service', '')}")
        if item.get('hierarchy'):
            print(f"📋 계층구조: {' > '.join(item.get('hierarchy', []))}")
        print(f"📊 종합 점수: {result['combined_score']:.4f}")
        
        # 대표 검색어 정보 출력
        representative_keywords = item.get('representative_keywords', [])
        if representative_keywords:
            print(f"🔍 AI 생성 대표 검색어: {', '.join(representative_keywords)}")
        
        print(f"{'='*60}")
    
    def generate_representative_keywords(self, item):
        """
        특정 메뉴 항목에 대해 OpenAI로부터 대표 검색어 5개 생성
        """
        try:
            category = item.get('Category', '')
            service = item.get('Service', '')
            page_name = item.get('page_name', '')
            hierarchy = ' > '.join(item.get('hierarchy', []))
            
            menu_info = f"카테고리: {category}, 서비스: {service}, 페이지명: {page_name}, 계층구조: {hierarchy}"
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 메뉴 검색 최적화 전문가입니다. 주어진 메뉴 정보를 보고, 사용자가 이 메뉴를 찾기 위해 검색할 가능성이 높은 검색어 5개를 생성해주세요. 각 검색어는 한국어로 작성하고, 쉼표로 구분해주세요. 실제 사용자들이 자연스럽게 사용할 법한 검색어를 만들어주세요."},
                    {"role": "user", "content": f"다음 메뉴에 대한 대표 검색어 5개를 생성해주세요:\n{menu_info}"}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            
            # 정확히 5개가 아니면 조정
            if len(keywords) > 5:
                keywords = keywords[:5]
            elif len(keywords) < 5:
                # 부족하면 기본 키워드들로 채움
                basic_keywords = [category, service, page_name]
                for bk in basic_keywords:
                    if bk and bk not in keywords and len(keywords) < 5:
                        keywords.append(bk)
            
            return keywords[:5]
            
        except Exception as e:
            print(f"❌ 대표 검색어 생성 실패 ({item.get('page_name', 'Unknown')}): {e}")
            # 실패시 기본 키워드 반환
            return [
                item.get('page_name', ''),
                item.get('Category', ''),
                item.get('Service', ''),
                ' '.join(item.get('hierarchy', [])),
                f"{item.get('Category', '')} {item.get('Service', '')}"
            ]
    
    def enhance_data_with_keywords(self):
        """
        메뉴 데이터에 대표 검색어를 추가하여 강화된 데이터 생성
        """
        # 이미 강화된 데이터가 있는지 확인
        if os.path.exists(self.enhanced_data_path):
            try:
                with open(self.enhanced_data_path, 'r', encoding='utf-8') as f:
                    enhanced_data = json.load(f)
                print(f"✅ 기존 강화된 데이터를 로드했습니다: {len(enhanced_data)}개 항목")
                return enhanced_data
            except Exception as e:
                print(f"⚠️ 기존 강화된 데이터 로드 실패: {e}")
        
        print(f"🤖 AI를 활용해 각 메뉴별 대표 검색어를 생성하고 있습니다...")
        print(f"📊 총 {len(self.data)}개 항목 처리 예정 (시간이 소요됩니다)")
        
        enhanced_data = []
        
        for i, item in enumerate(self.data, 1):
            print(f"🔄 진행중 [{i}/{len(self.data)}]: {item.get('page_name', 'Unknown')}")
            
            # 대표 검색어 생성
            representative_keywords = self.generate_representative_keywords(item)
            
            # 기존 데이터에 검색어 추가
            enhanced_item = item.copy()
            enhanced_item['representative_keywords'] = representative_keywords
            enhanced_data.append(enhanced_item)
            
            # API 호출 제한을 위한 딜레이
            if i % 10 == 0:
                print(f"⏱️ API 호출 제한 방지를 위해 잠시 대기중...")
                time.sleep(2)
            else:
                time.sleep(0.5)
        
        # 강화된 데이터 저장
        try:
            with open(self.enhanced_data_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 강화된 데이터를 저장했습니다: {self.enhanced_data_path}")
        except Exception as e:
            print(f"❌ 강화된 데이터 저장 실패: {e}")
        
        return enhanced_data


def main():
    """메인 함수"""
    print("🤖 AI 강화 메뉴 검색 프로그램")
    print("OpenAI API 키가 필요합니다.")
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("OpenAI API 키를 입력하세요: ").strip()
        if not api_key:
            print("❌ API 키가 필요합니다.")
            return
    
    try:
        searcher = AIMenuSearcher('ia-data.json', api_key)
        print(f"\n✅ 초기화 완료! 사용 모델: {searcher.current_model_name}")
        print("자연어로 검색어를 입력하세요 ('q' 입력시 종료):")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if query.lower() == 'q':
                    print("프로그램을 종료합니다.")
                    break
                
                if not query:
                    print("검색어를 입력해주세요.")
                    continue
                
                # AI 검색 수행
                result = searcher.ai_search(query)
                searcher.print_final_result(query, result)
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류가 발생했습니다: {e}")
    
    except Exception as e:
        print(f"❌ 초기화 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main() 