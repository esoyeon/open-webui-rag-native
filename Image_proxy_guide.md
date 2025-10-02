

# 🛠 Open WebUI 사내 이미지 표시 문제 해결 실무 가이드

## 1.  문제 정의

Open WebUI 환경에서 마크다운(![](url))을 사용해 이미지를 띄우려 할 때 다음과 같은 제약이 있음:
	•	사내망 이미지 서버 → 브라우저가 직접 접근 불가
(내부망 접근 제한, 인증 필요, CORS 문제 등)
	•	Base64 data URI → 너무 길고 성능 저하 발생
(메시지 잘림, 로딩 지연, 브라우저 메모리 부담)

➡️ 따라서, Open WebUI 백엔드가 대신 사내망 이미지를 가져와 브라우저로 전달하는 “프록시 엔드포인트” 방식이 필요함.

⸻

## 2. 해결 전략: 서버사이드 이미지 프록시
	•	Open WebUI 백엔드(FastAPI) 에 /api/proxy-image 라우트를 추가
	•	브라우저는 /api/proxy-image?src=... 로 접근
	•	백엔드가 사내 서버에서 이미지를 가져와 스트리밍으로 반환
	•	마크다운에서는 일반 이미지 URL처럼 사용 가능

⸻

## 3. 구현 단계

(1) 프록시 엔드포인트 코드 작성
src/backend/api/proxy.py 생성:
```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import httpx, urllib.parse, re

router = APIRouter()

# 허용 도메인 화이트리스트
SAFE_HOSTS = {"intranet-img.company.local", "media.internal"}

def is_safe_url(url: str) -> bool:
    try:
        u = urllib.parse.urlparse(url)
        return u.scheme in ("http", "https") and u.hostname in SAFE_HOSTS
    except:
        return False

@router.get("/proxy-image")
async def proxy_image(src: str, w: int | None = None, h: int | None = None):
    if not is_safe_url(src):
        raise HTTPException(status_code=400, detail="Blocked or invalid image src")

    async with httpx.AsyncClient(timeout=20.0) as client:
        upstream = await client.get(src)
        if upstream.status_code != 200:
            raise HTTPException(status_code=upstream.status_code, detail="Upstream error")

        ctype = upstream.headers.get("content-type", "application/octet-stream")
        if not re.match(r"^image/", ctype):
            raise HTTPException(status_code=400, detail="Not an image")

        return StreamingResponse(
            upstream.aiter_bytes(),
            media_type=ctype,
            headers={"Cache-Control": "public, max-age=86400"}
        )
```

⸻

(2) FastAPI 앱에 라우터 등록

src/backend/main.py 또는 백엔드 진입점에서:
```python
from fastapi import FastAPI
from .api import proxy

app = FastAPI()

# 기존 라우터들…
app.include_router(proxy.router, prefix="/api")
```

⸻

(3) 마크다운에서 사용하기
```
![샘플 이미지](/api/proxy-image?src=https%3A%2F%2Fintranet-img.company.local%2Ffiles%2Fsample.png)
```
	•	원본 URL을 urlencode 해서 src 파라미터로 넘김
	•	브라우저는 /api/proxy-image에 요청 → 백엔드가 내부망에서 가져와 스트리밍

⸻

## 4. 추가 기능 확장 (선택)
1.	썸네일 리사이즈	
		•	&w=800&h=600 옵션 지원 → Pillow로 리사이즈 후 반환
2.	인증 헤더/쿠키 주입
	
	```python
	upstream = await client.get(src, headers={"Authorization": "Bearer <TOKEN>"})
	```

3.	캐싱
		•	Cache-Control 헤더로 브라우저 캐시
		•	필요 시 서버 내부에 Redis/LRU 캐시 적용 가능
4.	보안 강화
		•	SAFE_HOSTS 화이트리스트로 SSRF 방지
		•	URL path 기반 ACL 가능

⸻

## 5. 테스트 절차
	
1.	Open WebUI 백엔드 실행 후
	```bash
	curl "http://localhost:3000/api/proxy-image?src=https%3A%2F%2Fintranet-img.company.local%2Ftest.png" -v
	```
	→ 응답 헤더에 Content-Type: image/png 확인

2.	Open WebUI 마크다운 입력창에
	```html
	![](/api/proxy-image?src=https%3A%2F%2Fintranet-img.company.local%2Ftest.png)
	```
	→ 이미지 정상 표시 확인

⸻

## 6. 운영 시 고려사항
- 로그 모니터링: 사내망 이미지 서버 응답 속도/오류율 추적
- 성능: 큰 이미지 반복 요청 시 서버 부하 증가 → 썸네일 + 캐시 권장
- 보안: SSRF 취약점 방지 위해 URL 필터링 필수

⸻

## ✅ 결론
- Base64 방식의 비효율을 제거하고,
- 내부망 이미지도 브라우저에서 쉽게 표시 가능
- /api/proxy-image 엔드포인트 추가만으로 해결

➡️ 사내 이미지 표시 문제는 “서버사이드 프록시 + 화이트리스트 보안” 패턴으로 안정적으로 해결할 수 있습니다.
