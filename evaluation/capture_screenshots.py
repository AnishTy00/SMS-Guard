"""Automated Browser Screenshot Capture for SMS Guard Application."""

import os
import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1400,950")

print("Initializing Edge WebDriver in headless mode...")
driver = webdriver.Edge(options=options)

try:
    url = "http://localhost:8501"
    print(f"Navigating to {url}...")
    driver.get(url)

    # Wait for Streamlit to render
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
    time.sleep(3)

    # 1. Capture Home Screen
    home_shot = os.path.join(EVAL_DIR, "screenshot_home.png")
    driver.save_screenshot(home_shot)
    print(f"[OK] Saved: {home_shot}")

    # 2. Click "Bank Phishing" preset button and Analyze
    buttons = driver.find_elements(By.TAG_NAME, "button")
    phishing_btn = None
    analyze_btn = None

    for btn in buttons:
        txt = btn.text
        if "Bank Phishing" in txt:
            phishing_btn = btn
        elif "Analyze Message" in txt:
            analyze_btn = btn

    if phishing_btn and analyze_btn:
        print("Clicking 'Bank Phishing' preset button...")
        phishing_btn.click()
        time.sleep(2)

        # Re-find analyze button after rerun
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if "Analyze Message" in btn.text:
                analyze_btn = btn
                break

        print("Clicking 'Analyze Message' button...")
        analyze_btn.click()
        time.sleep(3)

        result_shot = os.path.join(EVAL_DIR, "screenshot_analyzer_result.png")
        driver.save_screenshot(result_shot)
        print(f"[OK] Saved: {result_shot}")

    # 3. Click "Model Performance & Benchmarks" tab
    tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
    if len(tabs) >= 2:
        print("Navigating to Tab 2 (Model Performance & Benchmarks)...")
        tabs[1].click()
        time.sleep(3)
        perf_shot = os.path.join(EVAL_DIR, "screenshot_performance_tab.png")
        driver.save_screenshot(perf_shot)
        print(f"[OK] Saved: {perf_shot}")

    # 4. Click "Academic Foundations & Architecture" tab
    if len(tabs) >= 3:
        print("Navigating to Tab 3 (Academic Foundations & Architecture)...")
        tabs[2].click()
        time.sleep(3)
        acad_shot = os.path.join(EVAL_DIR, "screenshot_academic_tab.png")
        driver.save_screenshot(acad_shot)
        print(f"[OK] Saved: {acad_shot}")

finally:
    driver.quit()
    print("Browser closed. All screenshots captured successfully!")
