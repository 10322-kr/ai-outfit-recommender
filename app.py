"""인물 사진을 분석해 체형·분위기에 맞는 코디를 추천하는 Streamlit 앱 (Gemini API 사용)."""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import streamlit as st
from google import genai
from google.genai import errors, types
from PIL import Image
from pydantic import BaseModel, Field


class PersonAnalysis(BaseModel):
    has_person: bool = Field(description="이미지에 사람이 있는지 여부")
    body_shape: str = Field(description="체형·비율 특징과 장점 (긍정적이고 중립적인 표현으로)")
    skin_tone: str = Field(description="피부톤과 추정 퍼스널 컬러 (예: 웜톤 봄 추정)")
    vibe: str = Field(description="인상과 분위기")
    current_style: str = Field(description="현재 착장 요약")


class OutfitItem(BaseModel):
    category: str = Field(description="상의, 하의, 아우터, 원피스, 신발, 가방, 액세서리 중 하나")
    name: str = Field(description="구체적인 아이템명 (예: 크롭 니트 가디건, 하이웨이스트 와이드 슬랙스)")
    color: str = Field(description="추천 색상")


class Outfit(BaseModel):
    title: str = Field(description="코디 이름 (예: 꾸안꾸 주말 캐주얼)")
    items: list[OutfitItem] = Field(description="코디를 구성하는 아이템 4~6개")
    why_it_suits: str = Field(description="이 사람에게 어울리는 이유")
    styling_tip: str = Field(description="입는 법·연출 팁")


class Recommendation(BaseModel):
    person: PersonAnalysis
    best_colors: list[str] = Field(description="잘 어울리는 색상 4~6개")
    avoid: list[str] = Field(description="피하면 좋은 핏·색상·아이템 2~4개")
    outfits: list[Outfit] = Field(description="추천 코디 3세트")


BASE_PROMPT = """당신은 경력 많은 퍼스널 스타일리스트입니다.
사진 속 인물의 체형·비율, 피부톤, 인상과 분위기, 현재 착장을 분석하고
그 사람에게 가장 잘 어울리는 코디 3세트를 추천해 주세요.
- 체형은 장점을 살리는 관점에서 긍정적이고 존중하는 표현으로 설명하세요. 체중이나 외모를 평가하지 마세요.
- 각 코디는 서로 다른 분위기로 구성하고, 실제로 구매할 수 있는 구체적인 아이템명을 쓰세요.
- 사진에 사람이 없으면 has_person을 false로 두고 나머지는 일반적인 추천으로 채우세요.
모든 값은 한국어로 작성하세요."""

SEASONS = ["상관없음", "봄", "여름", "가을", "겨울"]
STYLES = ["미니멀", "캐주얼", "스트릿", "오피스/포멀", "러블리", "빈티지", "아메카지", "스포티", "클래식", "시티보이"]

# 우선순위 순서. 앞 모델의 하루 한도가 끝나면 다음 모델로 자동 전환한다.
# *-latest 별칭은 실제 모델과 한도를 공유하므로 넣지 않는다.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
DAILY_LIMIT_PER_MODEL = 20  # Gemini 무료 등급의 모델별 하루 요청 한도
DAILY_LIMIT_TOTAL = 100  # 공용 키의 하루 분석 횟수 상한 (개인 키는 상한 없음)

SHARED = "shared"  # 공용 키 사용량 소유자 ID
USAGE_FILE = Path(__file__).parent / "usage.json"
_usage_lock = threading.Lock()
_personal_usage: dict[str, dict] = {}  # 개인 키 사용량은 파일에 남기지 않고 서버 메모리에만 둔다 (키 해시별)


class QuotaExhaustedError(Exception):
    pass


def today() -> str:
    # 무료 한도는 태평양 시간 자정(한국 시간 오후 4~5시)에 초기화된다
    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def usage_owner(personal_key: str) -> str:
    return hashlib.sha256(personal_key.encode()).hexdigest()[:16] if personal_key else SHARED


def load_usage(owner: str) -> dict:
    if owner == SHARED:
        try:
            data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data = None
    else:
        data = _personal_usage.get(owner)
    if data and data.get("date") == today():
        return data
    return {"date": today(), "counts": {}, "exhausted": []}


def save_usage(owner: str, usage: dict) -> None:
    if owner == SHARED:
        USAGE_FILE.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        _personal_usage[owner] = usage


def total_used(usage: dict) -> int:
    return sum(usage["counts"].values())


def is_available(usage: dict, model: str) -> bool:
    return model not in usage["exhausted"] and usage["counts"].get(model, 0) < DAILY_LIMIT_PER_MODEL


def mark_exhausted(owner: str, model: str) -> None:
    with _usage_lock:
        usage = load_usage(owner)
        if model not in usage["exhausted"]:
            usage["exhausted"].append(model)
        save_usage(owner, usage)


def record_success(owner: str, model: str) -> None:
    with _usage_lock:
        usage = load_usage(owner)
        usage["counts"][model] = usage["counts"].get(model, 0) + 1
        save_usage(owner, usage)


def get_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # secrets.toml이 없으면 예외 발생
    return os.environ.get("GEMINI_API_KEY")


def build_prompt(season: str, styles: tuple[str, ...], extra: str) -> str:
    conditions = []
    if season != "상관없음":
        conditions.append(f"- 계절: {season}에 입기 좋은 소재와 레이어링으로 구성하세요.")
    if styles:
        conditions.append(f"- 원하는 스타일: {', '.join(styles)} 스타일을 중심으로 추천하세요.")
    if extra.strip():
        conditions.append(f"- 추가 요청: {extra.strip()}")
    if not conditions:
        return BASE_PROMPT
    return BASE_PROMPT + "\n\n[사용자 조건]\n" + "\n".join(conditions)


def call_gemini(client: genai.Client, model: str, image_bytes: bytes, mime_type: str, prompt: str) -> Recommendation:
    response = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Recommendation,
            temperature=0.7,
        ),
    )
    if response.parsed is None:
        return Recommendation.model_validate_json(response.text)
    return response.parsed


@st.cache_data(show_spinner=False)
def recommend(
    image_bytes: bytes, mime_type: str, prompt: str, api_key: str, owner: str
) -> tuple[Recommendation, str, list[str]]:
    """사용 가능한 모델을 우선순위대로 시도한다. (결과, 사용한 모델, 건너뛴 모델 목록)을 반환."""
    client = genai.Client(api_key=api_key)
    skipped = []
    for model in MODELS:
        usage = load_usage(owner)
        if owner == SHARED and total_used(usage) >= DAILY_LIMIT_TOTAL:
            raise QuotaExhaustedError(
                f"오늘 공용 분석 한도 {DAILY_LIMIT_TOTAL}회를 모두 사용했습니다. "
                "사이드바에 개인 API 키를 넣으면 계속 사용할 수 있습니다."
            )
        if not is_available(usage, model):
            continue
        try:
            result = call_gemini(client, model, image_bytes, mime_type, prompt)
        except errors.APIError as e:
            if (e.code == 429 and "PerDay" in str(e)) or e.code == 404:
                mark_exhausted(owner, model)  # 하루 한도 소진 또는 사용 불가 모델 → 오늘은 제외
                skipped.append(f"{model} (한도 소진)" if e.code == 429 else f"{model} (사용 불가)")
            elif e.code in (429, 500, 503):
                skipped.append(f"{model} (일시적 오류 {e.code})")  # 분당 한도·서버 혼잡 → 이번만 건너뜀
            else:
                raise
            continue
        record_success(owner, model)
        return result, model, skipped
    raise QuotaExhaustedError("지금 사용할 수 있는 모델이 없습니다. 잠시 후 다시 시도해 주세요.")


def render_usage(container, owner: str) -> None:
    usage = load_usage(owner)
    used = total_used(usage)
    with container.container():
        if owner == SHARED:
            st.subheader("오늘 공용 사용량")
            st.progress(min(used / DAILY_LIMIT_TOTAL, 1.0), text=f"{used} / {DAILY_LIMIT_TOTAL}회")
        else:
            st.subheader("오늘 개인 키 사용량")
            st.caption(f"총 {used}회 (공용 한도와 별개)")
        for model in MODELS:
            count = usage["counts"].get(model, 0)
            status = "✅" if is_available(usage, model) else "⛔"
            st.caption(f"{status} {model} — {count}/{DAILY_LIMIT_PER_MODEL}")
        st.caption("한도는 매일 태평양 시간 자정(한국 시간 오후 4~5시)에 초기화됩니다.")


def render_tags(tags: list[str], bg: str = "#eef2ff", fg: str = "#3730a3") -> None:
    html = " ".join(
        f"<span style='display:inline-block;padding:4px 12px;margin:3px;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:0.9rem'>{t}</span>"
        for t in tags
    )
    st.markdown(html, unsafe_allow_html=True)


SHOPS = {
    "지그재그": "https://zigzag.kr/search?keyword=",
    "무신사": "https://www.musinsa.com/search/goods?keyword=",
    "29CM": "https://www.29cm.co.kr/store/search?keyword=",
}


def shopping_links(item: OutfitItem) -> str:
    query = quote(f"{item.color} {item.name}")
    return " · ".join(f"[{name}]({url}{query})" for name, url in SHOPS.items())


MUSINSA_API = "https://api.musinsa.com/api2/dp/v2/plp/goods"
THUMB_SIZE = 96


def search_musinsa(color: str, name: str) -> dict | None:
    """무신사 검색 결과에서 아이템명과 가장 맞는 상품 하나를 고른다. 실패하면 None."""
    keyword = name.split()[-1]  # 예: '크롭 니트 가디건' → '가디건'
    for query in (f"{color} {name}", name):
        # requests 라이브러리는 무신사 봇 차단(403)에 걸려 urllib을 쓴다
        params = urlencode({"gf": "A", "keyword": query, "page": 1, "size": 10, "caller": "SEARCH"})
        request = Request(f"{MUSINSA_API}?{params}", headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=5) as resp:
                goods = json.load(resp)["data"]["list"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        candidates = [g for g in goods if g.get("thumbnail") and not g.get("isSoldOut") and not g.get("isAd")]
        if candidates:
            return next((g for g in candidates if keyword in g["goodsName"]), candidates[0])
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_products(items: tuple[tuple[str, str], ...]) -> dict[tuple[str, str], dict | None]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(items, pool.map(lambda i: search_musinsa(*i), items)))


def render_outfit(outfit: Outfit, products: dict) -> None:
    for item in outfit.items:
        product = products.get((item.color, item.name))
        col_thumb, col_text = st.columns([1, 5], vertical_alignment="center")
        with col_thumb:
            if product:
                st.markdown(
                    f"<a href='{product['goodsLinkUrl']}' target='_blank'>"
                    f"<img src='{product['thumbnail']}' width='{THUMB_SIZE}' "
                    f"style='border-radius:8px;aspect-ratio:1;object-fit:cover'></a>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='width:{THUMB_SIZE}px;height:{THUMB_SIZE}px;border-radius:8px;background:#f3f4f6;"
                    f"display:flex;align-items:center;justify-content:center;font-size:1.8rem'>👕</div>",
                    unsafe_allow_html=True,
                )
        with col_text:
            st.markdown(f"**{item.category}** · {item.color} {item.name}  \n🔎 {shopping_links(item)}")
            if product:
                price = f" · {product['price']:,}원" if isinstance(product.get("price"), int) else ""
                st.caption(f"예시: {product.get('brandName', '')} · {product['goodsName']}{price}")
    st.markdown(f"**💡 어울리는 이유** \n{outfit.why_it_suits}")
    st.markdown(f"**✨ 스타일링 팁** \n{outfit.styling_tip}")


st.set_page_config(page_title="AI 맞춤 코디 추천", page_icon="👗", layout="wide")
st.title("👗 AI 맞춤 코디 추천")
st.caption("인물 사진을 올리면 체형과 분위기에 어울리는 코디를 추천해 드립니다.")

with st.sidebar:
    st.header("설정")
    shared_key = get_api_key()
    personal_key = st.text_input(
        "내 Gemini API 키 (선택)",
        type="password",
        help="공용 한도(하루 100회)가 끝났거나 따로 쓰고 싶을 때 넣으세요. 키는 저장되지 않습니다.",
    ).strip()
    st.markdown("[무료 API 키 발급 (Google AI Studio)](https://aistudio.google.com/apikey)")
    if personal_key:
        st.success("개인 키로 분석합니다.")
    elif shared_key:
        st.caption("공용 키로 분석합니다.")
    api_key = personal_key or shared_key
    owner = usage_owner(personal_key)
    usage_slot = st.empty()
    render_usage(usage_slot, owner)

uploaded = st.file_uploader("전신 또는 상반신 인물 사진 업로드", type=["jpg", "jpeg", "png", "webp"])
st.caption(
    "🔒 업로드한 사진은 이 앱에 저장되지 않고, 분석을 위해 Google Gemini API로만 전송됩니다. "
    "무료 API 특성상 Google이 서비스 개선에 활용할 수 있으니 민감한 사진은 올리지 마세요."
)

if uploaded is None:
    st.info("추천받을 인물 사진을 업로드해 주세요. 전신 사진일수록 정확합니다.")
    st.stop()

col_img, col_result = st.columns([1, 1.6])
with col_img:
    st.image(Image.open(uploaded), use_container_width=True)

with col_result:
    if not api_key:
        st.warning("사이드바에 Gemini API 키를 입력해 주세요.")
        st.stop()

    c1, c2 = st.columns([1, 2])
    season = c1.selectbox("계절", SEASONS)
    styles = c2.multiselect("원하는 스타일 (선택)", STYLES)
    extra = st.text_input("추가 요청 (선택)", placeholder="예: 하객룩으로, 키가 커 보이게, 무채색 위주로")

    if st.button("✨ 코디 추천받기", type="primary", use_container_width=True):
        prompt = build_prompt(season, tuple(styles), extra)
        with st.spinner("Gemini가 어울리는 코디를 고르는 중입니다..."):
            try:
                result, used_model, skipped = recommend(uploaded.getvalue(), uploaded.type, prompt, api_key, owner)
            except QuotaExhaustedError as e:
                render_usage(usage_slot, owner)
                st.error(str(e))
                st.stop()
            except Exception as e:
                msg = str(e)
                if "API_KEY_INVALID" in msg or "401" in msg or "403" in msg:
                    st.error("API 키가 올바르지 않습니다.")
                else:
                    st.error(f"추천 중 오류가 발생했습니다: {msg}")
                st.stop()
        render_usage(usage_slot, owner)

        if skipped:
            st.info(f"{', '.join(skipped)} → **{used_model}** 모델로 분석했습니다.")
        else:
            st.caption(f"사용 모델: {used_model}")

        person = result.person
        if not person.has_person:
            st.warning("사진에서 인물을 찾지 못해 일반적인 코디를 추천했습니다.")
        else:
            with st.container(border=True):
                st.subheader("🧍 나의 스타일 분석")
                st.markdown(f"**체형·비율** \n{person.body_shape}")
                st.markdown(f"**피부톤·퍼스널 컬러** \n{person.skin_tone}")
                st.markdown(f"**분위기** \n{person.vibe}")
                st.markdown(f"**현재 착장** \n{person.current_style}")

        st.markdown("**🎨 잘 어울리는 색상**")
        render_tags(result.best_colors)
        st.markdown("**🚫 피하면 좋은 것**")
        render_tags(result.avoid, bg="#fef2f2", fg="#991b1b")

        st.subheader("👚 추천 코디")
        if result.outfits:
            with st.spinner("상품 이미지를 불러오는 중..."):
                products = fetch_products(tuple((i.color, i.name) for o in result.outfits for i in o.items))
            st.caption("이미지는 무신사 검색 결과의 예시 상품으로, 추천 아이템과 정확히 같지 않을 수 있습니다.")
            tabs = st.tabs([f"{i}. {o.title}" for i, o in enumerate(result.outfits, 1)])
            for tab, outfit in zip(tabs, result.outfits):
                with tab:
                    render_outfit(outfit, products)
        else:
            st.write("추천 코디를 만들지 못했습니다. 다시 시도해 주세요.")

        st.download_button(
            "추천 결과 JSON 다운로드",
            result.model_dump_json(indent=2),
            file_name="outfit_recommendation.json",
            mime="application/json",
        )
