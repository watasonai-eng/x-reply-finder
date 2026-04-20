import os
import sys
import io
import json
import time
from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import anthropic

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MY_PROFILE = os.getenv("MY_PROFILE", "")
KEYWORDS_RAW = os.getenv("KEYWORDS", "")
KEYWORDS = [k.strip() for k in KEYWORDS_RAW.split(",") if k.strip()]

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def stars(score: int) -> str:
    if score >= 80:
        return "★★★★★"
    elif score >= 60:
        return "★★★★☆"
    elif score >= 40:
        return "★★★☆☆"
    elif score >= 20:
        return "★★☆☆☆"
    else:
        return "★☆☆☆☆"


def search_tweets(driver, keyword: str) -> list[dict]:
    tweets = []
    try:
        search_url = f"https://x.com/search?q={keyword}&f=live"
        driver.get(search_url)

        # SPAのレンダリングを待機（最大15秒）
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='tweet']"))
            )
        except Exception:
            print(f"  ⚠️  ツイートの読み込みタイムアウト (URL: {driver.current_url})")
            return tweets

        collected_ids = set()
        scroll_attempts = 0
        max_scrolls = 10

        while len(tweets) < 20 and scroll_attempts < max_scrolls:
            articles = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
            print(f"  🔎 スクロール {scroll_attempts + 1}: article要素 {len(articles)} 件検出")
            for article in articles:
                if len(tweets) >= 20:
                    break
                try:
                    tweet_id = None
                    url = ""
                    username = ""
                    links = article.find_elements(By.CSS_SELECTOR, "a[href*='/status/']")
                    if links:
                        href = links[0].get_attribute("href")
                        if href and "/status/" in href:
                            parts = href.split("/")
                            status_idx = parts.index("status")
                            tweet_id = parts[status_idx + 1].split("?")[0]
                            username = parts[status_idx - 1] if status_idx > 0 else ""
                            url = "/".join(parts[:status_idx + 2])

                    if not tweet_id or tweet_id in collected_ids:
                        continue

                    text_els = article.find_elements(By.CSS_SELECTOR, "[data-testid='tweetText']")
                    text = text_els[0].text if text_els else ""

                    time_els = article.find_elements(By.TAG_NAME, "time")
                    posted_at = time_els[0].get_attribute("datetime") if time_els else ""

                    def get_metric(testid):
                        els = article.find_elements(By.CSS_SELECTOR, f"[data-testid='{testid}']")
                        if els:
                            txt = els[0].text.strip().replace(",", "").replace(".", "")
                            try:
                                return int(txt) if txt else 0
                            except ValueError:
                                return 0
                        return 0

                    likes = get_metric("like")
                    replies = get_metric("reply")
                    retweets = get_metric("retweet")

                    collected_ids.add(tweet_id)
                    tweets.append({
                        "id": tweet_id,
                        "username": username,
                        "text": text,
                        "likes": likes,
                        "replies": replies,
                        "retweets": retweets,
                        "posted_at": posted_at,
                        "url": url,
                        "keyword": keyword,
                    })
                except Exception as e:
                    print(f"    ⚠️  article解析エラー: {e}")
                    continue

            driver.execute_script("window.scrollBy(0, 1500)")
            # スクロール後にコンテンツが追加されるまで待機
            time.sleep(2)
            scroll_attempts += 1

    except Exception as e:
        print(f"  ⚠️  検索エラー ({keyword}): {e}")

    return tweets


SCORE_PROMPT_TEMPLATE = """あなたはXのリプ戦略の専門家です。
以下のアカウント情報と評価基準をもとに、各投稿にスコアをつけてください。

【アカウント情報】
{profile}

【評価基準（優先順）】
1. 投稿者のフォロワーにAI初心者・フリーランス・個人事業主が多そうか（最重要）
2. 投稿者のフォロワーに向けた補足リプが書けるか
3. 「フリーランス×AI活用」の文脈で自分の知見を自然に足せるか
4. 投稿から6時間以内か（鮮度。古いほど減点）
5. エンゲージメント率が高そうか

【即0点の条件】
- 架空ツール名・根拠不明な数値・誇大表現を含む投稿
- エンジニア・研究者向けの高度な技術投稿
- 炎上・政治・センシティブな話題
- 広告・宣伝色が強い投稿

各ツイートについて以下のJSON形式で返す：
[
  {{
    "id": "ツイートID",
    "score": 0-100の整数,
    "reason": "20文字以内の理由",
    "rep_angle": "リプの切り口（30文字以内）"
  }}
]
JSONのみ返し、前置き・説明文は不要。

【ツイート一覧】
{tweets_json}"""


def _call_api(tweet_summaries: list[dict]) -> list[dict]:
    prompt = SCORE_PROMPT_TEMPLATE.format(
        profile=MY_PROFILE,
        tweets_json=json.dumps(tweet_summaries, ensure_ascii=False, indent=2),
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            print("  ⚠️  JSONが見つかりませんでした")
            return []
        return json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSONパースエラー: {e}")
        return []
    except Exception as e:
        print(f"  ⚠️  APIエラー: {e}")
        return []


def score_tweets(tweets: list[dict]) -> list[dict]:
    tweet_summaries = [
        {
            "id": t["id"],
            "username": t["username"],
            "text": t["text"][:200],
            "likes": t["likes"],
            "replies": t["replies"],
            "retweets": t["retweets"],
            "posted_at": t["posted_at"],
        }
        for t in tweets
    ]

    batch_size = 20
    all_scores: list[dict] = []
    batches = [tweet_summaries[i:i + batch_size] for i in range(0, len(tweet_summaries), batch_size)]

    for i, batch in enumerate(batches, 1):
        print(f"  📦 バッチ {i}/{len(batches)} ({len(batch)}件) スコアリング中...")
        scores = _call_api(batch)
        all_scores.extend(scores)

    return all_scores


def display_results(ranked: list[dict]):
    for i, item in enumerate(ranked[:10], 1):
        tweet = item["tweet"]
        score_data = item["score_data"]
        score = score_data.get("score", 0)
        reason = score_data.get("reason", "")
        rep_angle = score_data.get("rep_angle", "")
        text = tweet["text"]
        preview = text[:50] + ("..." if len(text) > 50 else "")

        print("=" * 60)
        print(f" #{i}  スコア: {score}  {stars(score)}")
        print(f" @{tweet['username']}  ❤ {tweet['likes']}  💬 {tweet['replies']}  🔁 {tweet['retweets']}")
        print(f" 「{preview}」")
        print(f" 理由: {reason}")
        print(f" リプ角度: {rep_angle}")
        print(f" URL: {tweet['url']}")
    print("=" * 60)


def main():
    print("🔍 X リプおすすめ投稿ファインダー 起動中...")

    profile_path = os.path.abspath("./browser_profile")
    options = uc.ChromeOptions()
    options.add_argument(f"--user-data-dir={profile_path}")

    driver = uc.Chrome(options=options)

    try:
        driver.get("https://x.com")
        input("Xにログインしてください。完了したらEnterを押してください")

        while True:
            all_tweets: list[dict] = []
            seen_ids: set[str] = set()

            for idx, keyword in enumerate(KEYWORDS, 1):
                print(f"\n🔍 キーワード: [{keyword}] を検索中... ({idx}/{len(KEYWORDS)})")
                tweets = search_tweets(driver, keyword)
                for t in tweets:
                    if t["id"] not in seen_ids:
                        seen_ids.add(t["id"])
                        all_tweets.append(t)
                print(f"  → {len(tweets)} 件取得（重複除去後累計: {len(all_tweets)} 件）")

            if not all_tweets:
                print("⚠️  ツイートが収集できませんでした。")
            else:
                print(f"\n⏳ Claude APIでスコアリング中... (全{len(all_tweets)}件)")
                scores = score_tweets(all_tweets)

                score_map = {s["id"]: s for s in scores}
                ranked = []
                for t in all_tweets:
                    if t["id"] in score_map:
                        ranked.append({"tweet": t, "score_data": score_map[t["id"]]})

                ranked.sort(key=lambda x: x["score_data"].get("score", 0), reverse=True)

                print(f"\n📊 サマリー: 収集 {len(all_tweets)} 件 / スコアリング {len(ranked)} 件\n")
                display_results(ranked)

            answer = input("\nもう一度検索しますか？(y/n): ").strip().lower()
            if answer != "y":
                break

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
