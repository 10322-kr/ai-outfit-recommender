# 👗 AI 맞춤 코디 추천

인물 사진을 올리면 Gemini가 체형·분위기를 분석해 어울리는 코디 3세트를 추천하는 Streamlit 앱입니다.

## 로컬 실행

```
pip install -r requirements.txt
streamlit run app.py
```

공용 API 키는 `.streamlit/secrets.toml`에 넣습니다 (Git에 올리지 않음).

```toml
GEMINI_API_KEY = "발급받은 키"
```

## 배포 (Streamlit Community Cloud)

share.streamlit.io에서 이 저장소의 `app.py`를 선택해 배포하고,
Advanced settings → Secrets에 위와 같은 `GEMINI_API_KEY` 를 입력합니다.
