# 메뉴 검색 프로그램 🔍

벡터 임베딩을 통한 한국어 메뉴 검색 시스템

## 🚀 주요 기능

- **벡터 임베딩 검색**: 의미적 유사도 기반 검색
- **다중 모델 지원**: 5개 이상의 한국어 지원 모델
- **자동 모델 선택**: 성능 순서대로 자동 로드
- **TF-IDF 백업**: 모델이 없을 때 자동 전환

## 🏆 지원 모델

1. **multilingual-e5-small** 🥇 (최고 성능: 94.48% 유사도)
2. **ko-sroberta-multitask** 🥈 (한국어 특화: 85.23% 유사도)
3. **distiluse-base-multilingual-cased-v2** 🥉 (고속 처리)
4. **all-mpnet-base-v2** (영어 중심 고성능)
5. **paraphrase-multilingual-MiniLM-L12-v2** (다국어 지원)

## 📦 설치

```bash
# 의존성 설치
pip3 install -r requirements.txt

# 모델 다운로드 (선택사항)
python3 download_multiple_models.py
```

## 🎯 사용법

```bash
# 메뉴 검색 실행
python3 menu_search.py

# 모델 성능 비교
python3 compare_models.py
```

## 📊 검색 예시

```
검색어: 카드 이용내역

유사한 항목들:

1. 전체 유사도: 0.9725
   - 페이지별 유사도: 1.0000
   - 전체 컨텍스트 유사도: 0.8710
   - 종합 점수: 0.9448
   카테고리: 결제
   서비스: 카드관리
   페이지명: 카드관리
```

## 🛠 기술 스택

- **Python 3.13**
- **Sentence Transformers**: 벡터 임베딩
- **scikit-learn**: TF-IDF, 코사인 유사도
- **NumPy**: 수치 연산

## 📁 프로젝트 구조

```
menuSearch/
├── menu_search.py              # 메인 검색 프로그램
├── download_multiple_models.py # 모델 다운로드
├── compare_models.py           # 모델 성능 비교
├── requirements.txt            # 의존성
├── ia-data.json               # 검색 데이터
├── models/                    # 다운로드된 모델들 (Git 제외)
└── README.md                  # 이 파일
```

## 🎨 특징

- **실시간 검색**: 터미널에서 즉시 검색
- **고정확도**: 최고 94.48% 유사도
- **다국어**: 한국어 포함 50개 언어 지원
- **자동화**: 최적 모델 자동 선택

## 📈 성능

| 모델 | 평균 유사도 | 로드 시간 | 임베딩 시간 |
|------|-------------|-----------|-------------|
| multilingual-e5-small | 94.48% | 1.24초 | 0.414초 |
| ko-sroberta-multitask | 85.23% | 0.38초 | 1.858초 |
| manual_model | 81.80% | 1.14초 | 0.046초 |

---

*Made with ❤️ by AI Assistant* 