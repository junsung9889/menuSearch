#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 메뉴 검색 시스템
- 자동 페이지 분류 + 데이터 강화 + 하이브리드 검색
- 스마트 캐싱으로 중복 작업 방지  
- OpenAI API와 최신 AI 기술을 결합한 차세대 검색 플랫폼
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import os
from openai import OpenAI
import time
import re
from collections import Counter
import math

warnings.filterwarnings("ignore")

# SSL 인증서 문제 해결
import ssl
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


class UnifiedMenuSearcher:
    def __init__(self, openai_api_key=None):
        """통합 메뉴 검색기 초기화"""
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        
        # 파일 경로 정의
        self.original_data_file = 'ia-data.json'
        self.filtered_data_file = 'ia-data_filtered.json'
        self.enhanced_data_file = 'ia-data_enhanced.json'
        
        # 단계별 데이터 로드
        self.original_data = self._load_original_data()
        self.filtered_data = self._load_or_create_filtered_data()
        self.enhanced_data = self._load_or_create_enhanced_data()
        
        # 모델 로드
        self._load_model()
        
        # 검색 인덱스 생성
        if not self.use_tfidf:
            self.embeddings = self._create_embeddings()
        else:
            self._setup_tfidf()
        
        self._setup_bm25()
    
    def _load_original_data(self):
        """원본 데이터 로드"""
        try:
            with open(self.original_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 원본 데이터 로드: {len(data)}개 항목")
            return data
        except Exception as e:
            print(f"❌ 원본 데이터 로드 실패: {e}")
            return []
    
    def _load_or_create_filtered_data(self):
        """필터링된 데이터 로드 또는 생성"""
        if os.path.exists(self.filtered_data_file):
            try:
                with open(self.filtered_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ 기존 필터링된 데이터 로드: {len(data)}개 항목")
                return data
            except Exception as e:
                print(f"⚠️ 필터링된 데이터 로드 실패: {e}")
        
        print("🔍 필터링된 데이터를 생성합니다...")
        return self._create_filtered_data()
    
    def _load_or_create_enhanced_data(self):
        """강화된 데이터 로드 또는 생성"""
        if os.path.exists(self.enhanced_data_file):
            try:
                with open(self.enhanced_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ 기존 강화된 데이터 로드: {len(data)}개 항목")
                return data
            except Exception as e:
                print(f"⚠️ 강화된 데이터 로드 실패: {e}")
        
        print("🚀 강화된 데이터를 생성합니다...")
        return self._create_enhanced_data()
    
    def _create_filtered_data(self):
        """페이지 분류를 통한 필터링된 데이터 생성"""
        print("🔧 규칙 기반 페이지 분류 시작...")
        
        # 규칙 기반 필터링 패턴
        exclude_keywords = [
            "오류", "실패", "없음", "empty", "안내", "동의", "브릿지", "점검중",
            "세션", "타임아웃", "미연결", "업데이트", "스플래시", "랜딩", "복호화",
            "탐지", "장애", "종료", "재실행", "권한", "스켈레톤"
        ]
        
        error_patterns = [
            r".*오류.*", r".*실패.*", r".*없음.*", r".*empty.*", r".*error.*",
            r".*fail.*", r".*timeout.*", r".*세션.*", r".*점검.*"
        ]
        
        bridge_patterns = [
            r".*브릿지.*", r".*bridge.*", r".*랜딩.*", r".*landing.*",
            r".*스플래시.*", r".*splash.*"
        ]
        
        filtered_data = []
        
        for item in self.original_data:
            page_name = item.get('page_name', '').strip()
            category = item.get('Category', '')
            service = item.get('Service', '')
            hierarchy = item.get('hierarchy', [])
            
            # 공백 페이지 제외
            if not page_name or page_name.isspace():
                continue
            
            # 오류 페이지 제외
            is_error = any(re.match(pattern, page_name, re.IGNORECASE) for pattern in error_patterns)
            if is_error:
                continue
            
            # 브릿지 페이지 제외
            is_bridge = any(re.match(pattern, page_name, re.IGNORECASE) for pattern in bridge_patterns)
            if is_bridge:
                continue
            
            # 키워드 기반 제외
            is_excluded = any(keyword in page_name.lower() or keyword in service.lower() for keyword in exclude_keywords)
            if is_excluded:
                continue
            
            # 계층 깊이 체크 (너무 깊은 것은 제외)
            if len(hierarchy) > 5:
                continue
            
            # 유의미한 페이지로 판단되면 추가
            enhanced_item = item.copy()
            enhanced_item['is_meaningful'] = True
            enhanced_item['filter_reason'] = 'rules_based'
            filtered_data.append(enhanced_item)
        
        # 저장
        try:
            with open(self.filtered_data_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 필터링된 데이터 저장: {self.filtered_data_file} ({len(filtered_data)}개 항목)")
        except Exception as e:
            print(f"❌ 필터링된 데이터 저장 실패: {e}")
        
        return filtered_data
    
    def _create_enhanced_data(self):
        """AI를 활용한 강화된 데이터 생성"""
        print("🤖 AI를 활용해 데이터를 강화합니다...")
        print(f"📊 총 {len(self.filtered_data)}개 항목 처리 예정")
        
        enhanced_data = []
        
        for i, item in enumerate(self.filtered_data, 1):
            print(f"🔄 진행중 [{i}/{len(self.filtered_data)}]: {item.get('page_name', 'Unknown')}")
            
            enhanced_item = item.copy()
            
            # 대표 검색어 생성
            enhanced_item['representative_keywords'] = self._generate_representative_keywords(item)
            
            # AI 설명 생성
            enhanced_item['ai_description'] = self._generate_ai_description(item)
            
            # 페이지 분류
            enhanced_item['page_classification'] = self._classify_page_importance(item)
            
            enhanced_data.append(enhanced_item)
            
            # API 호출 제한을 위한 딜레이
            if i % 10 == 0:
                print(f"⏱️ API 호출 제한 방지를 위해 잠시 대기중...")
                time.sleep(2)
            else:
                time.sleep(0.5)
        
        # 저장
        try:
            with open(self.enhanced_data_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 강화된 데이터 저장: {self.enhanced_data_file}")
        except Exception as e:
            print(f"❌ 강화된 데이터 저장 실패: {e}")
        
        return enhanced_data
    
    def _generate_representative_keywords(self, item):
        """대표 검색어 생성"""
        try:
            category = item.get('Category', '')
            service = item.get('Service', '')
            page_name = item.get('page_name', '')
            hierarchy = ' > '.join(item.get('hierarchy', []))
            
            menu_info = f"카테고리: {category}, 서비스: {service}, 페이지명: {page_name}, 계층구조: {hierarchy}"
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 메뉴 검색 최적화 전문가입니다. 주어진 메뉴 정보를 보고, 사용자가 이 메뉴를 찾기 위해 검색할 가능성이 높은 검색어 5개를 생성해주세요. 각 검색어는 한국어로 작성하고, 쉼표로 구분해주세요."},
                    {"role": "user", "content": f"다음 메뉴에 대한 대표 검색어 5개를 생성해주세요:\n{menu_info}"}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            return keywords[:5]
            
        except Exception as e:
            print(f"❌ 대표 검색어 생성 실패: {e}")
            return [page_name, category, service]
    
    def _generate_ai_description(self, item):
        """AI 설명 생성"""
        try:
            category = item.get('Category', '')
            service = item.get('Service', '')
            page_name = item.get('page_name', '')
            hierarchy = ' > '.join(item.get('hierarchy', []))
            
            menu_info = f"카테고리: {category}, 서비스: {service}, 페이지명: {page_name}, 계층구조: {hierarchy}"
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 메뉴 설명 전문가입니다. 주어진 메뉴 정보를 보고, 사용자가 이해하기 쉬운 한 문장의 설명을 만들어주세요. '이 페이지는 ~를 할 수 있는 화면입니다' 형태로 작성해주세요."},
                    {"role": "user", "content": f"다음 메뉴에 대한 설명을 생성해주세요:\n{menu_info}"}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ AI 설명 생성 실패: {e}")
            return f"이 페이지는 {item.get('page_name', '')} 관련 기능을 제공하는 화면입니다."
    
    def _classify_page_importance(self, item):
        """페이지 중요도 분류"""
        page_name = item.get('page_name', '')
        
        # 핵심 기능 키워드
        primary_keywords = ['홈', '카드', '결제', '관리', '설정', '조회', '내역', '명세서', '포인트']
        
        if any(keyword in page_name for keyword in primary_keywords):
            return 'PRIMARY'
        else:
            return 'SECONDARY'
    
    def _load_model(self):
        """모델 로드"""
        local_model_paths = [
            "./models/multilingual-e5-small",
            "./models/ko-sroberta-multitask",
            "./models/distiluse-base-multilingual-cased-v2",
            "./models/all-mpnet-base-v2",
        ]
        
        model_loaded = False
        self.current_model_name = "Unknown"
        
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
    
    def _create_enhanced_text_for_embedding(self, item):
        """강화된 임베딩용 텍스트 생성"""
        category = item.get('Category', '')
        service = item.get('Service', '')
        page_name = item.get('page_name', '')
        hierarchy = ' > '.join(item.get('hierarchy', []))
        
        basic_text = f"카테고리: {category} 서비스: {service} 페이지명: {page_name} 계층구조: {hierarchy}"
        
        # 대표 검색어 가중치
        representative_keywords = item.get('representative_keywords', [])
        if representative_keywords:
            keywords_text = ' '.join(representative_keywords)
            weighted_keywords = f" {keywords_text} {keywords_text} {keywords_text}"
        else:
            weighted_keywords = ""
        
        # AI 설명 가중치
        ai_description = item.get('ai_description', '')
        if ai_description:
            weighted_description = f" {ai_description} {ai_description}"
        else:
            weighted_description = ""
        
        return basic_text + weighted_keywords + weighted_description
    
    def _create_embeddings(self):
        """강화된 벡터 임베딩 생성"""
        print("🔮 강화된 벡터 임베딩을 생성하고 있습니다...")
        
        enhanced_texts = [self._create_enhanced_text_for_embedding(item) for item in self.enhanced_data]
        page_texts = [item.get('page_name', '') for item in self.enhanced_data]
        ai_descriptions = [item.get('ai_description', '') for item in self.enhanced_data]
        
        embeddings = {
            'enhanced': self.model.encode(enhanced_texts, show_progress_bar=True),
            'page': self.model.encode(page_texts, show_progress_bar=True),
            'description': self.model.encode(ai_descriptions, show_progress_bar=True)
        }
        
        print("✅ 강화된 벡터 임베딩 생성 완료")
        return embeddings
    
    def _setup_tfidf(self):
        """TF-IDF 설정"""
        texts = [self._create_enhanced_text_for_embedding(item) for item in self.enhanced_data]
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def _setup_bm25(self):
        """BM25 인덱스 설정"""
        print("🔍 BM25 키워드 검색 인덱스를 생성하고 있습니다...")
        
        bm25_texts = []
        for item in self.enhanced_data:
            text_parts = [
                item.get('page_name', ''),
                item.get('Category', ''),
                item.get('Service', ''),
                ' '.join(item.get('hierarchy', [])),
                ' '.join(item.get('representative_keywords', []))
            ]
            bm25_text = ' '.join(filter(None, text_parts))
            bm25_texts.append(bm25_text)
        
        self.bm25 = BM25(bm25_texts)
        print("✅ BM25 인덱스 생성 완료")
    
    def hybrid_search(self, query, top_k=7, vector_weight=0.7, keyword_weight=0.3):
        """하이브리드 검색: 벡터 + 키워드"""
        results = []
        
        if self.use_tfidf:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            for i, similarity in enumerate(similarities):
                results.append({
                    'index': i,
                    'data': self.enhanced_data[i],
                    'vector_score': float(similarity),
                    'keyword_score': 0.0,
                    'hybrid_score': float(similarity)
                })
        else:
            query_embedding = self.model.encode([query])
            vector_similarities = cosine_similarity(query_embedding, self.embeddings['enhanced'])[0]
            keyword_scores = self.bm25.get_scores(query)
            
            max_vector_score = max(vector_similarities) if max(vector_similarities) > 0 else 1
            max_keyword_score = max(keyword_scores) if max(keyword_scores) > 0 else 1
            
            for i, (vector_sim, keyword_score) in enumerate(zip(vector_similarities, keyword_scores)):
                normalized_vector = vector_sim / max_vector_score
                normalized_keyword = keyword_score / max_keyword_score
                
                hybrid_score = (normalized_vector * vector_weight) + (normalized_keyword * keyword_weight)
                
                results.append({
                    'index': i,
                    'data': self.enhanced_data[i],
                    'vector_score': float(vector_sim),
                    'keyword_score': float(keyword_score),
                    'hybrid_score': float(hybrid_score)
                })
        
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:top_k]
    
    def ai_final_selection(self, original_query, candidates):
        """AI 최종 선택"""
        try:
            menu_text = ""
            for i, result in enumerate(candidates, 1):
                item = result['data']
                ai_description = item.get('ai_description', '')
                menu_text += f"""{i}. {item.get('page_name', '')}
   카테고리: {item.get('Category', '')}
   서비스: {item.get('Service', '')}
   설명: {ai_description}
   대표검색어: {', '.join(item.get('representative_keywords', []))}
   
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """당신은 사용자의 의도를 정확히 파악해 가장 적합한 메뉴를 선택하는 전문가입니다. 
사용자의 원래 질문과 각 메뉴의 상세 정보를 종합적으로 고려하여, 가장 관련성이 높고 사용자가 원하는 기능을 제공할 수 있는 메뉴 하나의 번호만 답하세요. 
반드시 숫자만 답하세요."""},
                    {"role": "user", "content": f"""사용자 질문: '{original_query}'

메뉴 후보들:
{menu_text}

가장 적합한 메뉴의 번호를 선택하세요:"""}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            selected_num = int(response.choices[0].message.content.strip())
            if 1 <= selected_num <= len(candidates):
                selected_menu = candidates[selected_num - 1]
                print(f"🎯 AI 최종 선택: {selected_num}번 - {selected_menu['data'].get('page_name', '')}")
                return selected_menu
            else:
                return candidates[0]
                
        except Exception as e:
            print(f"❌ AI 메뉴 선택 실패: {e}")
            return candidates[0]
    
    def search(self, query):
        """통합 검색 수행"""
        print(f"\n{'='*60}")
        print(f"🚀 통합 AI 검색 시작: '{query}'")
        print(f"{'='*60}")
        
        # 1단계: 하이브리드 검색
        print(f"\n🔍 1단계: 하이브리드 검색 (벡터 + 키워드)")
        candidates = self.hybrid_search(query, top_k=7)
        
        if not candidates:
            print("❌ 검색 결과가 없습니다.")
            return None
        
        print(f"   찾은 후보 {len(candidates)}개:")
        for i, result in enumerate(candidates, 1):
            item = result['data']
            page_name = item.get('page_name', '')
            vector_score = result['vector_score']
            keyword_score = result['keyword_score']
            hybrid_score = result['hybrid_score']
            
            print(f"   {i}. {page_name}")
            print(f"      벡터: {vector_score:.3f} | 키워드: {keyword_score:.3f} | 통합: {hybrid_score:.3f}")
        
        # 2단계: AI 최종 선택
        print(f"\n🎯 2단계: AI 최종 선택")
        final_result = self.ai_final_selection(query, candidates)
        
        return final_result
    
    def print_result(self, query, result):
        """검색 결과 출력"""
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
        
        print(f"📊 하이브리드 점수: {result.get('hybrid_score', 0):.4f}")
        print(f"   ┣ 벡터 점수: {result.get('vector_score', 0):.4f}")
        print(f"   ┗ 키워드 점수: {result.get('keyword_score', 0):.4f}")
        
        ai_description = item.get('ai_description', '')
        if ai_description:
            print(f"🤖 AI 설명: {ai_description}")
        
        representative_keywords = item.get('representative_keywords', [])
        if representative_keywords:
            print(f"🔍 대표 검색어: {', '.join(representative_keywords)}")
        
        page_classification = item.get('page_classification', '')
        if page_classification:
            print(f"📈 페이지 분류: {page_classification}")
        
        print(f"{'='*60}")
    
    def get_statistics(self):
        """시스템 통계 정보"""
        print(f"\n{'='*60}")
        print(f"📊 시스템 통계")
        print(f"{'='*60}")
        print(f"📄 원본 데이터: {len(self.original_data)}개 항목")
        print(f"✨ 필터링된 데이터: {len(self.filtered_data)}개 항목")
        print(f"🚀 강화된 데이터: {len(self.enhanced_data)}개 항목")
        print(f"🧠 사용 모델: {self.current_model_name}")
        
        # 분류 통계
        primary_count = sum(1 for item in self.enhanced_data if item.get('page_classification') == 'PRIMARY')
        secondary_count = len(self.enhanced_data) - primary_count
        
        print(f"🎯 PRIMARY 페이지: {primary_count}개")
        print(f"📋 SECONDARY 페이지: {secondary_count}개")
        print(f"📈 필터링 효율성: {((len(self.original_data) - len(self.filtered_data)) / len(self.original_data) * 100):.1f}% 제거")
        print(f"{'='*60}")


def main():
    """메인 함수"""
    print("🚀 AI 메뉴 검색 시스템")
    print("자동 필터링 + 데이터 강화 + 하이브리드 검색")
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("OpenAI API 키를 입력하세요: ").strip()
        if not api_key:
            print("❌ API 키가 필요합니다.")
            return
    
    try:
        # 시스템 초기화
        searcher = UnifiedMenuSearcher(api_key)
        
        # 통계 정보 출력
        searcher.get_statistics()
        
        print("\n자연어로 검색어를 입력하세요 ('q' 입력시 종료, 'stats' 입력시 통계 조회):")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if query.lower() == 'q':
                    print("프로그램을 종료합니다.")
                    break
                
                if query.lower() == 'stats':
                    searcher.get_statistics()
                    continue
                
                if not query:
                    print("검색어를 입력해주세요.")
                    continue
                
                # 검색 수행
                result = searcher.search(query)
                searcher.print_result(query, result)
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류가 발생했습니다: {e}")
    
    except Exception as e:
        print(f"❌ 초기화 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 