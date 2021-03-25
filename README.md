# Kakao Arena Brunch Posts Recommendation System

## Contents-based Recommender System
### Post Feature: TF-IDF
- `metadata`의 태그 정보를 바탕으로 TF-IDF 행렬을 생성하여 글 각각의 feature vector를 얻음
- `TF` type: boolean(글에 해당 태그가 등장할 경우 1, 그렇지 않을 경우 0)
- `IDF` type: logarithm
```python
>>> python3 calculate_export_tfidf.py tfidf --root-dir './raw' --vocab-path './tfidf'
>>> python3 calculate_export_tfidf.py df --root-dir './raw' --vocab-path './tfidf'
```

### User Preference
- 유저 각각이 과거에 조회했던 글들의 feature vector를 바탕으로 유저의 feature vector를 얻음
- 유저가 일정 기간동안 조회한 글들의 TF-IDF 벡터를 가중합
    - DF를 가중치로 활용
```python
>>> python3 user_preference.py run --user-id-list 'dev' --start 2019022200 --end 2019022200 --save-path './dev_recommendation_sources/'
```

### Recommendation
- 유저 feature vector에 가장 부합하는 글들을 추천
- 유저 feature vector와 각 글들의 feature vector를 내적, 값이 가장 높은 순서대로 100개의 글을 추천
- 단, 유저가 이미 조회한 글은 추천에서 제외
```python
>>> python3 contents_based_rs.py run --recommend-src './recommendation_sources/2018100100-2019022200' --output-root './output'

### Ongoing
![](https://github.com/iloveslowfood/KakaoBrunchRS/blob/main/images/structure.jpg?raw=true)

### Task
- 브런치 유저가 2019년 2월 22일 이후로 볼 글 100개를 추천
- 다음과 같은 형태로 예측
```
1 @wo-motivator133(유저 아이디) @wo-motivator134(추천 글 1) …
2 @backcharcruz34 @artsbz23 …
```


### Data Structure

![data_structure](https://github.com/iloveslowfood/KakaoBrunchRS/blob/main/images/data_structure.jpg?raw=true)

#### read: 본 글 정보
- read.tar: 2018년 10월 1일부터 2019년 3월 1일까지 브런치 독자 일부가 본 글 정보가 3,625개의 파일로 구성
- 파일명 포맷: `시작일_종료일`
    - `2018110708_2018110709`: 2018년 11월 7일 오전 8시부터 2018년 11월 7일 오전 9시 전까지 본 글
- 파일은 여러 줄로 이뤄져 있으며, 한 줄은 브런치의 독자가 파일의 시간 동안 본 글을 시간 순으로 기록한 것
    - 한 줄 정보는 공백으로 구분, 첫째는 독자의 암호화된 식별자, 그 뒤로는 해당 독자가 본 글 정보
    - 예: `read/2019022823_2019030100` 파일의  
    `#8a706ac921a11004bab941d22323efab @bakchacruz_34 @wo-motivator_133 @wo-motivator_133`
    - `#8a706ac921a11004bab941d22323efab` 독자가 2019년 2월 28일 23시에서 2019년 3월 1일 0시 사이에 `@bakchacruz_34` `@wo-motivator_133` `@wo-motivator_133` 를 순서대로 보았다는 뜻
    - @wo-motivator_133 글이 두 번 나타난 것은 이 글을 보기 위해 두 번 방문했다는 뜻
    - '글을 보았다': 특정 글에 모바일, PC, 앱을 통해 접근했다
        - 머문 시간 정보는 제공되지 않아, 글을 읽지 않고 이탈했을 가능성 O
        
#### metadata.json: 글의 메타데이터
- 643,104 줄로 구성
- 2018년 10월 1일부터 2019년 3월 14일까지 독자들이 본 글에 대한 정보
- 작가가 비공개로 전환하였거나 삭제 등의 이유로 이 메타데이터에 없는 글이 있을 수 있음
- 개발 데이터와 평가 데이터에 포함된 글의 메타데이터도 포함
    - 즉, 평가 대상자들이 2019년 3월 1일부터 2019년 3월 14일 사이에 본 모든 글 정보 포함
- 필드 설명
    - `magazine_id`: 이 글의 브런치 매거진 아이디 (없을 시는 0)
    - `reg_ts`: 이 글이 등록된 시간(유닉스 시간, 밀리초)
    - `user_id`: 작가 아이디
    - `article_id`: 글 번호
    - `id`: 글 식별자
    - `title`: 제목
    - `sub_title`: 부제목
    - `display_url`: 웹 주소
    - `keyword_list`: 작가가 부여한 글의 태그 정보
- 메타데이터의 모든 정보는 작가의 비공개 여부 전환, 글 삭제, 수정 등으로 유효하지 않거나 변동될 수 있음

#### contents: 글 본문 정보
- 저작권을 보호하고자 본문에서 형태소 분석을 통해 추출된 정보를 암호화하여 제공
- 총 7개의 파일
- 형태소 분석: 카카오 `khaiii` 의 기본 옵션을 사용
    - 형태소 분석 결과의 어휘 정보는 임의의 숫자로 1:1 변환
    - 동일 어휘의 경우, 품사와 관계없이 같은 숫자로 변환
    - 형태소 분석에 대한 설명과 품사의 의미에 대해서는 별도 제공하지 않음
    - 형태소 추출 전에 텍스트를 제외한 HTML과 같은 내용과 관계없는 정보는 제거 했으나 일부 정보가 남았을 수 있음
- 필드 설명
    - `id`: 글 식별자
    - `morphs`: 형태소 분석 결과
        - 리스트의 리스트로 구성되며, 리스트의 첫 번째 요소는 첫 어절의 분석 결과
        - 어휘와 품사는 `/` 구분자로 구분됩니다.
        - 예: "안녕하세요 브런치입니다"
            - khaiii 형태소분석기에서 "안녕/NNG + 하/XSA + 시/EP + 어요/EF", "브런치/NNP + 이/VCP + ㅂ니다/EC" 라고 분석되고, `morphs`에 다음과 같이 저장
            - `[["8/NNG", "13/XSA", "81/EP", "888/EF"], ["0/NNP", "12913/VCP", "29/EC"]]`
        - 여러 줄에 걸친 결과는 개행 구분 없이 리스트에 연속적으로 등장
        - 예: "안녕하세요 브런치입니다\n안녕하세요"의 형태소 분석 결과는 다음과 같음
            - `[["8/NNG", "13/XSA", "81/EP", "888/EF"], ["0/NNP", "12913/VCP", "29/EC"], ["8/NNG", "13/XSA", "81/EP", "888/EF"]]`
    - `chars`: 형태소 분석 결과
        - 형태소 분석 결과에서 어휘 부분을 문자 단위로 암호화환 결과
        - 한 어휘의 문자는 `+` 구분자로 결합
            - 예: "브런치입니다"는 `chars` 필드에서 다음과 같이 표현
            - `"0+1+2/NNP", "4/VCP", "9+29+33/EC"`
- `metadata.json`과 마찬가지로 개발 데이터와 평가 데이터의 글 본문도 포함
- `contents` 정보는 본문이 없는 글의 경우 제공되지 않을 수 있음

#### users.json: 사용자 정보
- 가입한 사용자(작가 혹은 독자) 310,758명의 정보
    - 탈퇴 등의 이유로 사용자 정보가 없을 수 있음
- 필드 설명
    - `keyword_list`: 최근 며칠간 작가 글로 유입되었던 검색 키워드
    - `following_list`: 구독 중인 작가 리스트
    - `id`: 사용자 식별자

#### magazine.json: 매거진 정보
- 27,967개 브런치 매거진 정보
- 필드 설명
    - `id`: 매거진 식별자
    - `magazine_tag_list`: 작가가 부여한 매거진의 태그 정보

#### predict 디렉토리: 예측할 사용자 정보
- `dev.users`: 개발 데이터. 대회 기간에 예측한 성능 평가를 위해 제공한 사용자 3,000명 리스트
- `test.users`: 평가 데이터입니다. 대회 종료 후 최종 순위 결정을 위해 제공한 사용자 5,000명 리스트
- 일부 사용자는 2018년 10월 1일부터 2019년 3월 1일까지 본 글이 없을 수도 있음

## etc

#### Magazine

- 브런치의 비슷한 주제로 작성한 글들이 모인 공간
- '작가' 자격을 가진 사용자가 '매거진 만들기'를 통해 새로운 매거진을 만들 수 있음
- 새로운 매거진을 만들 때 **매거진의 태그(`megazine_tag_list`)** 추가 가능
- '참여 신청'을 통해 다른 작가가 해당 매거진에 합류할 수 있음
- Reference. https://brunch.co.kr/@brunch/4

