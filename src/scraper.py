import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

from path import *

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
}

def parse_card(card):
    try:
        # make and model from the URL
        href = card.get('href', '')
        parts = href.strip('/').split('/')
        manufacturer = parts[4].replace('-', ' ').title()
        model = parts[5].replace('-', ' ').title()

        # price
        price_el = card.select_one('.offer_card__price_value')
        price = int(re.sub(r'[^\d]', '', price_el.text)) if price_el else None

        # desc (engine liters, horsepower, fuel type and transmission)
        desc_el = card.select_one('.offer_card__desc')
        desc = desc_el.text.strip() if desc_el else ''

        engine_match = re.search(r'(\d+\.\d+)', desc)
        hp_match = re.search(r'\((\d+)\s*hp\)', desc)
        parts_desc = [p.strip() for p in desc.split(',')]

        engine_liters = float(engine_match.group(1)) if engine_match else None
        horsepower = int(hp_match.group(1)) if hp_match else None
        fuel = parts_desc[1] if len(parts_desc) > 1 else None
        transmission = parts_desc[2] if len(parts_desc) > 2 else None

        # hand and year
        meta = card.select('.offer_card__meta li')
        year = int(meta[0].text.strip()) if meta else None
        hand_text = meta[1].text.strip() if len(meta) > 1 else ''
        hand_match = re.search(r'(\d+)', hand_text)
        hand = int(hand_match.group(1)) if hand_match else None

        return {
            'manufacturer': manufacturer,
            'model': model,
            'year': year,
            'price': price,
            'hand': hand,
            'fuel': fuel,
            'engine_liters': engine_liters,
            'horsepower': horsepower,
            'transmission': transmission,
        }

    except Exception as e:
        print(f'eror: {e}')
        return None


def get_submodel(ad_url):
    try:
        response = requests.get(ad_url, headers=HEADERS, timeout=30)
        ad_soup = BeautifulSoup(response.text, 'html.parser')

        offer_subheader = ad_soup.select_one('.offer_subheader')
        if offer_subheader:
            text = offer_subheader.get_text(strip=False)
            text = text.split(' ')
            text = [x for x in text if x != '']
            text = text[2:]
            result = " ".join(text)
        return result if result.strip() else None
    except Exception as e:
        print(f'eror: {e}')
        return None


def get_mileage(ad_url):
    try:
        response = requests.get(ad_url, headers=HEADERS, timeout=30)
        ad_soup = BeautifulSoup(response.text, 'html.parser')

        # looking for the span wite the speedometer icon
        speed_icon = ad_soup.select_one('.u_icon-speed_indicator')
        if speed_icon:
            # going to the parent and finding the text
            param_div = speed_icon.find_parent('div')
            if param_div:
                text = param_div.get_text(strip=True)
                km_match = re.search(r'([\d\xa0\s]+)km', text)
                if km_match:
                    return int(re.sub(r'[^\d]', '', km_match.group(1)))

        return None

    except Exception as e:
        print(f'eror: {e}')
        return None


def scrape_page(manufacturer, page):
    url = f'https://autoboom.co.il/en/used/cars/{manufacturer}?year_from=2015&page={page}'
    response = requests.get(url, headers=HEADERS, timeout=60)
    
    if response.status_code != 200:
        print(f'eror in page {page}: status {response.status_code}')
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.select('.offer_card__visible')
    
    results = []
    for link in links:
        card = parse_card(link)
        if not card:
            continue
        
        ad_url = 'https://autoboom.co.il' + link.get('href')
        card['mileage'] = get_mileage(ad_url)
        card['submodel'] = get_submodel(ad_url)
        results.append(card)
        time.sleep(0.5)
    
    return results


MANUFACTURERS = ['kia', 'toyota', 'hyundai', 'skoda', 'mazda', 'nissan', 'chevrolet', 'honda', 'mitsubishi', 'peugeot', 'suzuki', 'audi', 'ford', 'subaru']
#  
PAGES_PER_MANUFACTURER = 30 

def scrape_all():
    all_results = []

    for manufacturer in MANUFACTURERS:
        print(f'\n===== {manufacturer.upper()} =====')
        manufacturer_results = []

        for page in range(1, PAGES_PER_MANUFACTURER + 1):
            print(f'page {page}/{PAGES_PER_MANUFACTURER}...')
            page_results = scrape_page(manufacturer, page)
            manufacturer_results.extend(page_results)
            print(f'so far {len(manufacturer_results)} ads')
            time.sleep(1)

        # saving csv for each manufacturer
        df = pd.DataFrame(manufacturer_results)
        out_path = os.path.join(DATA_DIR, f'{manufacturer}_data.csv')
        df.to_csv(out_path, index=False)
        print(f'saved as: {manufacturer}_data.csv')

        all_results.extend(manufacturer_results)

    # cimbining all the data
    df_all = pd.DataFrame(all_results)
    out = os.path.join(DATA_DIR, 'autoboom_raw.csv')
    df_all.to_csv(out, index=False)
    print(f'\nin total: {len(df_all)} adds saved in - autoboom_raw.csv')
    return df_all



df = scrape_all()
print(df.head())