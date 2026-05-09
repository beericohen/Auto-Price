import re
import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from path import *

MANUFACTURERS = ['kia', 'toyota', 'hyundai', 'skoda', 'mazda', 'nissan', 'chevrolet', 'honda', 'mitsubishi', 'peugeot', 'suzuki', 'audi', 'ford', 'subaru']
PAGES_PER_MANUFACTURER = 30


def create_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36')
    return webdriver.Chrome(options=options)


def get_param(soup, label_text):
    params = soup.select('.offer_param')
    for param in params:
        label = param.select_one('.offer_param__label')
        value = param.select_one('.offer_param__value')
        if label and value and label_text.lower() in label.get_text().lower():
            result = value.get_text(strip=True)
            return result if result else None
    return None


def get_ad_links(driver, manufacturer, page):
    url = f'https://autoboom.co.il/en/used/cars/{manufacturer}?year_from=2015&page={page}'
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.offer_card__visible'))
        )
    except:
        return []
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = soup.select('.offer_card__visible')
    return ['https://autoboom.co.il' + l.get('href', '') for l in links if l.get('href')]


def scrape_ad(driver, url):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.offer_info__price'))
        )

        # grab price, year, hand, mileage BEFORE clicking the button
        soup_before = BeautifulSoup(driver.page_source, 'html.parser')

        # price
        price_el = soup_before.select_one('.offer_info__price span')
        price = int(re.sub(r'[^\d]', '', price_el.get_text())) if price_el else None

        # mileage
        mileage = None
        speed_icon = soup_before.select_one('.u_icon-speed_indicator')
        if speed_icon:
            param_div = speed_icon.find_parent('div')
            if param_div:
                km_match = re.search(r'([\d\xa0\s]+)km', param_div.get_text(strip=True))
                if km_match:
                    mileage = int(re.sub(r'[^\d]', '', km_match.group(1)))

        # year
        year = None
        year_el = soup_before.select_one('.u_icon-calendar')
        if year_el:
            year_div = year_el.find_parent('div')
            if year_div:
                year_match = re.search(r'(\d{4})', year_div.get_text())
                year = int(year_match.group(1)) if year_match else None

        # hand
        hand = None
        hand_el = soup_before.select_one('.u_icon-hand')
        if hand_el:
            hand_div = hand_el.find_parent('div')
            if hand_div:
                hand_match = re.search(r'(\d+)', hand_div.get_text())
                hand = int(hand_match.group(1)) if hand_match else None

        # click "Show all data" button
        try:
            btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.base_block__show_all'))
            )
            btn.click()
            time.sleep(1)
        except:
            pass  # some ads may not have the button

        soup_after = BeautifulSoup(driver.page_source, 'html.parser')

        # make and model from URL
        parts = url.strip('/').split('/')
        manufacturer = parts[-3].replace('-', ' ').title()
        model = parts[-2].replace('-', ' ').title()

        # engine liters, horsepower, fuel from engine param e.g. "1.4 l / 100 hp / Gasoline"
        engine_str = get_param(soup_after, 'Engine')
        engine_liters, horsepower, fuel = None, None, None
        if engine_str:
            engine_match = re.search(r'(\d+\.\d+)', engine_str)
            hp_match = re.search(r'(\d+)\s*hp', engine_str)
            fuel_match = re.search(r'hp\s*/\s*(\w+)', engine_str)
            engine_liters = float(engine_match.group(1)) if engine_match else None
            horsepower    = int(hp_match.group(1))       if hp_match     else None
            fuel          = fuel_match.group(1)           if fuel_match   else None

        return {
            'manufacturer' : manufacturer,
            'model'        : model,
            'submodel'     : get_param(soup_after, 'Trim version'),
            'year'         : year,
            'hand'         : hand,
            'mileage'      : mileage,
            'engine_liters': engine_liters,
            'horsepower'   : horsepower,
            'fuel'         : fuel,
            'transmission' : get_param(soup_after, 'Transmission'),
            'drive_type'   : get_param(soup_after, 'Type of drive'),
            'body_color'   : get_param(soup_after, 'Body color'),
            'price'        : price,
        }

    except Exception as e:
        print(f'error scraping {url}: {e}')
        return None


def scrape_all():
    all_results = []
    driver = create_driver()  # one driver for the entire scrape

    try:
        for manufacturer in MANUFACTURERS:
            print(f'\n===== {manufacturer.upper()} =====')
            manufacturer_results = []

            for page in range(1, PAGES_PER_MANUFACTURER + 1):
                print(f'page {page}/{PAGES_PER_MANUFACTURER}...')

                ad_links = get_ad_links(driver, manufacturer, page)
                print(f'found {len(ad_links)} ads on page')

                for ad_url in ad_links:
                    result = scrape_ad(driver, ad_url)
                    if result:
                        manufacturer_results.append(result)
                    time.sleep(0.5)

                print(f'so far {len(manufacturer_results)} ads')
                time.sleep(1)

            # save csv per manufacturer
            df = pd.DataFrame(manufacturer_results)
            out_path = os.path.join(DATA_DIR, f'{manufacturer}_data.csv')
            df.to_csv(out_path, index=False)
            print(f'saved as: {manufacturer}_data.csv')

            all_results.extend(manufacturer_results)

    finally:
        driver.quit()  # always close the browser

    # combine all data
    df_all = pd.DataFrame(all_results)
    out = os.path.join(DATA_DIR, 'autoboom_raw.csv')
    df_all.to_csv(out, index=False)
    print(f'\nin total: {len(df_all)} ads saved in autoboom_raw.csv')
    return df_all


df = scrape_all()
print(df.head())