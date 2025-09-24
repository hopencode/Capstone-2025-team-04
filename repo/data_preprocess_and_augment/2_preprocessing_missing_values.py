import pandas as pd

def preprocess_and_clean_data(file_path, output_file_path=None):
    # 데이터 로드 & 결측치 처리
    try:
        df = pd.read_csv(
            file_path,
            dtype={'zip': str}
        )
        df.replace({'zip': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)
        df.replace({'merchant_state': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)
        df.replace({'errors': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)
        df.replace({'fraud': {'': pd.NA, 'NULL': pd.NA}}, inplace=True)
    except Exception as e:
        print(f"파일 로딩 오류: {e}")
        return None

    print("데이터 로드 완료.\n")

    # 'merchant_city'가 'ONLINE'인 경우 처리
    online_cond = df['merchant_city'] == 'ONLINE'
    df.loc[online_cond & df['merchant_state'].isna(), 'merchant_state'] = 'ONLINE'
    if 'zip' in df.columns:
        df.loc[online_cond & df['zip'].isna(), 'zip'] = '00000'
    print("온라인 결제 zip 처리 완료")


    # 미국 주 목록
    us_states = [
        'AA','AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA',
        'ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT',
        'NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC',
        'SD','TN','TX','UT','VA','VT','WA','WI','WV','WY'
    ]
    intl_cond = (df['merchant_state'].notna()) & (~df['merchant_state'].isin(us_states)) & (df['merchant_city'] != 'ONLINE')

    # 해외 국가 결제의 ZIP 처리
    if 'zip' in df.columns:
        intl_zip_na = intl_cond & df['zip'].isna()
        country_zip_map = {
            'MEXICO': 'MEX00', 'VATICAN CITY': 'VAT00', 'DOMINICAN REPUBLIC': 'DOM00',
            'CANADA': 'CAN00', 'ALBANIA': 'ALB00', 'ALGERIA': 'ALG00', 'ANDORRA': 'AND00',
            'ARGENTINA': 'ARG00', 'ARUBA': 'ARU00', 'AUSTRALIA': 'AUS00',
            'AUSTRIA': 'AUT00', 'AZERBAIJAN': 'AZE00', 'BAHRAIN': 'BAH00',
            'BANGLADESH': 'BGD00', 'BARBADOS': 'BRB00', 'BELGIUM': 'BEL00',
            'BELIZE': 'BLZ00', 'BENIN': 'BEN00', 'BOSNIA AND HERZEGOVINA': 'BIH00',
            'BRAZIL': 'BRA00', 'BRUNEI': 'BRN00', 'BURKINA FASO': 'BFA00',
            'CABO VERDE': 'CPV00', 'CAMEROON': 'CMR00', 'CHILE': 'CHL00',
            'CHINA': 'CHN00', 'COLOMBIA': 'COL00', 'COSTA RICA': 'CRI00',
            'COTE D IVOIRE': 'CIV00', 'CROATIA': 'HRV00', 'CYPRUS': 'CYP00',
            'CZECH REPUBLIC': 'CZE00', 'DENMARK': 'DNK00', 'EAST TIMOR (TIMOR-LESTE)': 'TLS00',
            'ECUADOR': 'ECU00', 'EGYPT': 'EGY00', 'EQUATORIAL GUINEA': 'GNQ00',
            'ERITREA': 'ERI00', 'ESTONIA': 'EST00', 'ETHIOPIA': 'ETH00',
            'FIJI': 'FJI00', 'FINLAND': 'FIN00', 'FRANCE': 'FRA00',
            'GABON': 'GAB00', 'GEORGIA': 'GEO00', 'GERMANY': 'DEU00',
            'GHANA': 'GHA00', 'GREECE': 'GRC00', 'GUATEMALA': 'GTM00',
            'GUINEA': 'GIN00', 'GUYANA': 'GUY00', 'HAITI': 'HTI00',
            'HONDURAS': 'HND00', 'HONG KONG': 'HKG00', 'HUNGARY': 'HUN00',
            'ICELAND': 'ISL00', 'INDIA': 'IND00', 'INDONESIA': 'IDN00',
            'IRAN': 'IRN00', 'IRAQ': 'IRQ00', 'IRELAND': 'IRL00',
            'ISRAEL': 'ISR00', 'ITALY': 'ITA00', 'JAMAICA': 'JAM00',
            'JAPAN': 'JPN00', 'JORDAN': 'JOR00', 'KENYA': 'KEN00',
            'KOSOVO': 'KOS00', 'KYRGYZSTAN': 'KGZ00', 'LATVIA': 'LVA00',
            'LEBANON': 'LBN00', 'LIBERIA': 'LBR00', 'LITHUANIA': 'LTU00',
            'LUXEMBOURG': 'LUX00', 'MACEDONIA': 'MKD00', 'MALAYSIA': 'MYS00',
            'MALDIVES': 'MDV00', 'MALI': 'MLI00', 'MALTA': 'MLT00',
            'MARSHALL ISLANDS': 'MHL00', 'MICRONESIA': 'FSM00', 'MOLDOVA': 'MDA00',
            'MONACO': 'MCO00', 'MONGOLIA': 'MNG00', 'MONTENEGRO': 'MNE00',
            'MOROCCO': 'MAR00', 'MOZAMBIQUE': 'MOZ00', 'MYANMAR (BURMA)': 'MMR00',
            'NAURU': 'NRU00', 'NETHERLANDS': 'NLD00', 'NEW ZEALAND': 'NZL00',
            'NIGER': 'NER00', 'NIGERIA': 'NGA00', 'NORWAY': 'NOR00',
            'OMAN': 'OMN00', 'PAKISTAN': 'PAK00', 'PANAMA': 'PAN00',
            'PAPUA NEW GUINEA': 'PNG00', 'PERU': 'PER00', 'PHILIPPINES': 'PHL00',
            'POLAND': 'POL00', 'PORTUGAL': 'PRT00', 'QATAR': 'QAT00',
            'REPUBLIC OF THE CONGO': 'COG00', 'ROMANIA': 'ROU00', 'RUSSIA': 'RUS00',
            'SAINT VINCENT AND THE GRENADINES': 'VCT00', 'SAMOA': 'WSM00',
            'SAUDI ARABIA': 'SAU00', 'SENEGAL': 'SEN00', 'SERBIA': 'SRB00',
            'SEYCHELLES': 'SYC00', 'SIERRA LEONE': 'SLE00', 'SINGAPORE': 'SGP00',
            'SLOVAKIA': 'SVK00', 'SLOVENIA': 'SVN00', 'SOLOMON ISLANDS': 'SLB00',
            'SOUTH AFRICA': 'ZAF00', 'SOUTH KOREA': 'KOR00', 'SOUTH SUDAN': 'SSD00',
            'SPAIN': 'ESP00', 'SRI LANKA': 'LKA00', 'SUDAN': 'SDN00',
            'SURINAME': 'SUR00', 'SWAZILAND': 'SWZ00', 'SWEDEN': 'SWE00',
            'SWITZERLAND': 'CHE00', 'TAIWAN': 'TWN00', 'THAILAND': 'THA00',
            'THE BAHAMAS': 'BHS00', 'TONGA': 'TON00', 'TRINIDAD AND TOBAGO': 'TTO00',
            'TUNISIA': 'TUN00', 'TURKEY': 'TUR00', 'TUVALU': 'TUV00',
            'UKRAINE': 'UKR00', 'UNITED ARAB EMIRATES': 'ARE00', 'UNITED KINGDOM': 'GBR00',
            'URUGUAY': 'URY00', 'UZBEKISTAN': 'UZB00', 'VANUATU': 'VUT00',
            'VENEZUELA': 'VEN00', 'VIETNAM': 'VNM00', 'YEMEN': 'YEM00',
            'ZAMBIA': 'ZMB00', 'ZIMBABWE': 'ZWE00'
        }
        def assign_foreign_zip(row):
            if pd.isna(row['zip']) and intl_zip_na[row.name]:
                state_upper = str(row['merchant_state']).upper().strip()
                return country_zip_map.get(state_upper, 'OTH00')
            return row['zip']
        df['zip'] = df.apply(assign_foreign_zip, axis=1)
    print("해외 결제 zip 처리 완료")

    # 모든 zip 값을 5자리 문자열로 표준화
    if 'zip' in df.columns:
        def format_zip(zip_val):
            if pd.isna(zip_val):
                return pd.NA
            str_val = str(zip_val).strip()
            try:
                return str(int(float(str_val))).zfill(5)
            except ValueError:
                return str_val
        df['zip'] = df['zip'].apply(format_zip)
    print("zip 결측치 처리 완료.\n")

    # 'errors' 결측값 'No Error'로 채우기
    if 'errors' in df.columns:
        df['errors'] = df['errors'].fillna('No Error')
    print("errors 결측치 처리 완료.\n")

    # 'fraud' 결측값 있는 행 삭제
    if 'fraud' in df.columns:
        df.dropna(subset=['fraud'], inplace=True)
    print("fraud 결측치 처리 완료.\n")

    # 저장
    if output_file_path:
        try:
            df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"저장 오류: {e}")
    print("저장 완료.\n")

    return df

if __name__ == "__main__":
    transactions_data_input = './data/transactions_fraud_label.csv'
    transactions_data_output = './data/transactions_fraud_label_preprocess.csv'
    processed_df = preprocess_and_clean_data(transactions_data_input, transactions_data_output)

    if processed_df is not None:
        print(processed_df[['merchant_city','merchant_state','zip','errors','fraud']].head(20))
