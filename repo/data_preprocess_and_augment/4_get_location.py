import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# 미국 주 목록 (참고용)
us_states_and_territories = [
    'AA', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
    'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI',
    'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY',
    'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT',
    'WA', 'WI', 'WV', 'WY', 'PR', 'VI', 'GU', 'AS', 'MP'
]

# Geocoding 서비스 초기화
geolocator = Nominatim(user_agent="merchant_location_extractor")


def get_location_from_zip(zip_code):
    try:
        location = geolocator.geocode({"postalcode": zip_code, "country": "USA"})
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Error fetching location for zip {zip_code}: {e}")
        time.sleep(1)  # API 요청 제한 방지
    return None, None


def get_location_from_city_state(city, state):
    query = f"{city}, {state}"
    try:
        location = geolocator.geocode(query, addressdetails=True)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Error fetching location for {query}: {e}")
        time.sleep(1)
    return None, None



def main():
    # 1. 거래 데이터 로드 (실제 데이터 파일 경로로 변경 필요)
    df = pd.read_csv('./data/augmented_transaction.csv', dtype={'zip': str})

    # 2. 필요한 컬럼 추출 및 중복 제거
    locations_df = df[['zip', 'merchant_state', 'merchant_city']].drop_duplicates().reset_index(drop=True)

    # 위도/경도 컬럼 추가
    locations_df['latitude'] = None
    locations_df['longitude'] = None

    # 3. 각 상점 위치에 대한 좌표 조회
    for index, row in locations_df.iterrows():
        zip_code = str(row['zip']).zfill(5)
        merchant_state = str(row['merchant_state'])
        merchant_city = str(row['merchant_city'])

        if zip_code == '00000' or merchant_state.upper() == 'ONLINE':
            locations_df.loc[index, 'latitude'] = np.nan
            locations_df.loc[index, 'longitude'] = np.nan
            continue

        latitude, longitude = None, None

        # 4. 미국 지역 상점인 경우 zip으로 좌표 조회 시도
        if merchant_state.upper() in us_states_and_territories:
            latitude, longitude = get_location_from_zip(zip_code)

        # 5. zip으로 조회 실패 시 또는 미국 외 지역인 경우 city, state로 조회 시도
        if latitude is None or longitude is None:
            latitude, longitude = get_location_from_city_state(merchant_city, merchant_state)

        # 6. 결과 저장
        locations_df.loc[index, 'latitude'] = latitude
        locations_df.loc[index, 'longitude'] = longitude

        # API 요청 제한 방지를 위해 딜레이 추가
        time.sleep(0.5)
    locations_df = locations_df.sort_values(by='zip', ascending=True).reset_index(drop=True)

    # 7. CSV 파일로 저장
    output_filename = './data/merchant_locations.csv'
    locations_df.to_csv(output_filename, index=False)
    print(f"상점 위치 정보가 '{output_filename}' 파일에 성공적으로 저장되었습니다.")
    print("\n저장된 파일의 일부 내용은 다음과 같습니다.")
    print(locations_df.head())


if __name__ == "__main__":
    main()
