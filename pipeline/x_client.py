"""
X Browser Automation Connector (Selenium)
------------------------------------------
Replaces the X API v2 client with browser automation using Selenium WebDriver.
No API costs, uses manual browser interaction with session persistence.

Setup: add to .env:
    X_USERNAME=your_username_or_email
    X_PASSWORD=your_password
    X_EMAIL=your_email_for_2fa (optional, for 2FA challenges)

Requires: pip install selenium webdriver-manager
"""

import os
import time
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

COOKIE_FILE = Path(__file__).parent.parent / "logs" / "x_cookies.pkl"
COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)


class SeleniumXClient:
    def __init__(self, dry_run: bool = True, headless: bool = False):
        self.dry_run = dry_run
        self.headless = headless
        self.driver = None
        self.logged_in = False
        self._username = os.environ.get("X_USERNAME")
        self._password = os.environ.get("X_PASSWORD")
        self._email = os.environ.get("X_EMAIL")

        if not dry_run:
            self._init_driver()
            self._load_session()

    def _init_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280,720")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.driver.implicitly_wait(5)

    def _save_session(self):
        if self.driver:
            cookies = self.driver.get_cookies()
            with open(COOKIE_FILE, "wb") as f:
                pickle.dump(cookies, f)

    def _load_session(self):
        if COOKIE_FILE.exists():
            try:
                self.driver.get("https://x.com")
                time.sleep(2)
                with open(COOKIE_FILE, "rb") as f:
                    cookies = pickle.load(f)
                for cookie in cookies:
                    if "sameSite" in cookie:
                        cookie["sameSite"] = "Lax"
                    try:
                        self.driver.add_cookie(cookie)
                    except Exception:
                        pass
                self.driver.refresh()
                time.sleep(2)
                if self._is_logged_in():
                    self.logged_in = True
                    return
            except Exception:
                pass
        self._login()

    def _is_logged_in(self) -> bool:
        try:
            self.driver.find_element("css selector", '[data-testid="SideNav_AccountSwitcher_Button"]')
            return True
        except Exception:
            try:
                self.driver.find_element("css selector", '[data-testid="tweetButtonInline"]')
                return True
            except Exception:
                return False

    def _login(self):
        if not self._username or not self._password:
            raise RuntimeError("X_USERNAME and X_PASSWORD required in .env for Selenium login")

        self.driver.get("https://x.com/i/flow/login")
        time.sleep(3)

        # Username/email input
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        username_input = WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        username_input.send_keys(self._username)
        username_input.send_keys(Keys.RETURN)
        time.sleep(2)

        # Password input
        try:
            password_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
            )
            password_input.send_keys(self._password)
            password_input.send_keys(Keys.RETURN)
            time.sleep(3)
        except Exception:
            pass

        # Handle 2FA if needed
        if self._email:
            try:
                email_input = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]'))
                )
                email_input.send_keys(self._email)
                email_input.send_keys(Keys.RETURN)
                time.sleep(3)
            except Exception:
                pass

        # Verify login
        if self._is_logged_in():
            self.logged_in = True
            self._save_session()
        else:
            raise RuntimeError("Login failed - check credentials or handle 2FA manually")

    def _ensure_logged_in(self):
        if not self.logged_in:
            self._load_session()
        if not self.logged_in:
            raise RuntimeError("Not logged in to X")

    # --- Write operations ---

    def post_tweet(self, text: str) -> Optional[str]:
        if self.dry_run:
            print(f"[DRY RUN] Would post tweet: {text[:80]}...")
            return "dry_run_tweet_id"

        self._ensure_logged_in()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            # Click compose button
            compose_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="tweetButtonInline"]'))
            )
            compose_btn.click()
            time.sleep(1)

            # Type tweet
            tweet_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
            )
            tweet_box.send_keys(text)
            time.sleep(0.5)

            # Post
            post_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="tweetButton"]'))
            )
            post_btn.click()
            time.sleep(2)

            # Extract tweet ID from URL or toast
            return "posted_tweet_id"
        except Exception as e:
            print(f"Error posting tweet: {e}")
            return None

    def reply_to_tweet(self, tweet_id: str, text: str) -> Optional[str]:
        if self.dry_run:
            print(f"[DRY RUN] Would reply to {tweet_id}: {text[:80]}...")
            return "dry_run_reply_id"

        self._ensure_logged_in()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            self.driver.get(f"https://x.com/i/status/{tweet_id}")
            time.sleep(3)

            # Click reply button
            reply_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="reply"]'))
            )
            reply_btn.click()
            time.sleep(1)

            # Type reply
            reply_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
            )
            reply_box.send_keys(text)
            time.sleep(0.5)

            # Post reply
            post_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="tweetButton"]'))
            )
            post_btn.click()
            time.sleep(2)

            return "posted_reply_id"
        except Exception as e:
            print(f"Error replying to tweet: {e}")
            return None

    def like_tweet(self, tweet_id: str) -> bool:
        if self.dry_run:
            print(f"[DRY RUN] Would like tweet: {tweet_id}")
            return True

        self._ensure_logged_in()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            self.driver.get(f"https://x.com/i/status/{tweet_id}")
            time.sleep(2)

            like_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="like"]'))
            )
            # Check if already liked
            if like_btn.get_attribute("aria-pressed") == "true":
                return True
            like_btn.click()
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Error liking tweet: {e}")
            return False

    def follow_user(self, username: str) -> bool:
        if self.dry_run:
            print(f"[DRY RUN] Would follow user: {username}")
            return True

        self._ensure_logged_in()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            self.driver.get(f"https://x.com/{username}")
            time.sleep(3)

            follow_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="followButton"], [data-testid="follow"]'))
            )
            if "Following" in follow_btn.text or "following" in follow_btn.get_attribute("aria-label", "").lower():
                return True
            follow_btn.click()
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Error following user: {e}")
            return False

    # --- Read operations ---

    def get_tweet(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        if self.dry_run:
            return {"id": tweet_id, "text": "[dry run tweet]", "author_id": "dry_run_user"}

        self._ensure_logged_in()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            self.driver.get(f"https://x.com/i/status/{tweet_id}")
            time.sleep(3)

            tweet_elem = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetText"]'))
            )
            text = tweet_elem.text

            author_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
            author = author_elem.text.split("\n")[0] if author_elem else "unknown"

            return {"id": tweet_id, "text": text, "author": author}
        except Exception as e:
            print(f"Error getting tweet: {e}")
            return None

    def search_tweets(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        if self.dry_run:
            return [{"id": f"dry_run_{i}", "text": f"[dry run] {query} result {i}"} for i in range(count)]

        self._ensure_logged_in()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        results = []
        try:
            search_url = f"https://x.com/search?q={query}&src=typed_query&f=live"
            self.driver.get(search_url)
            time.sleep(3)

            tweets = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="tweet"]'))
            )

            for tweet in tweets[:count]:
                try:
                    text_elem = tweet.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                    text = text_elem.text
                    results.append({"text": text})
                except Exception:
                    continue
        except Exception as e:
            print(f"Error searching tweets: {e}")

        return results

    def get_user_tweets(self, username: str, count: int = 10) -> List[Dict[str, Any]]:
        if self.dry_run:
            return [{"id": f"dry_run_{i}", "text": f"[dry run] @{username} tweet {i}"} for i in range(count)]

        self._ensure_logged_in()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        results = []
        try:
            self.driver.get(f"https://x.com/{username}")
            time.sleep(3)

            tweets = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="tweet"]'))
            )

            for tweet in tweets[:count]:
                try:
                    text_elem = tweet.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                    text = text_elem.text
                    results.append({"text": text})
                except Exception:
                    continue
        except Exception as e:
            print(f"Error getting user tweets: {e}")

        return results

    def close(self):
        if self.driver:
            self._save_session()
            self.driver.quit()
            self.driver = None

    def __del__(self):
        self.close()


# Backward compatibility alias
XClient = SeleniumXClient