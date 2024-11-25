import requests
import time
import pandas as pd
import re
from scrapy import Selector


def cik_company_tickers(user_mail: str, deduplicate = False) -> pd.DataFrame:

    """
    This function returns all tickers for cik entities which are currently listed by the SEC.

    Parameters:
    -----------
    user_mail: the mail which is used as a header information as demanded by the SEC

    Example:
    -----------
    """

    headers = {"User-Agent": f"{user_mail}"}
    url = "https://www.sec.gov/files/company_tickers.json"

    r = requests.get(url, headers=headers)
    time.sleep(.1)
    tickers_cik = pd.json_normalize(pd.json_normalize(r.json(), max_level=0).values[0])
    tickers_cik["cik"] = tickers_cik["cik_str"].astype(str).str.zfill(10)
    tickers_cik.drop('cik_str', axis=1, inplace=True)
    if deduplicate:
        # deduplicate tickers which correspond to the same cik (company)
        tickers_cik = tickers_cik.groupby('cik').agg({'ticker': ','.join, 
                                    'title':'first' }).reset_index()
    tickers_cik = tickers_cik[['cik', 'ticker', 'title']]
    tickers_cik.loc[:, "as_of_date"] = pd.Timestamp.today()
    return tickers_cik


def get_filing_history(cik: str, user_mail: str):

    """
    This function returns the filing history for a given cik including meta data.

    Parameters:
    -----------
    cik: cik identifier used by the SEC
    user_mail: the mail which is used as a header information which is demanded by the SEC
    """

    headers = {"User-Agent": f"{user_mail}"}

    if len(cik) <= 10:
        cik = cik.zfill(10)
    elif len(cik) > 10:
        raise ValueError("cik is an identifier with a maximum of ten characters")

    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    r = requests.get(url, headers = headers)
    time.sleep(.1)
    filings = pd.DataFrame(r.json()['filings']['recent'])
    # if more than 1000 recent filings exist, they can be found under the files key
    for extra_filings in r.json()['filings']['files']:
        url_tmp = f'https://data.sec.gov/submissions/' + extra_filings['name']
        r_tmp = requests.get(url_tmp, headers = headers)
        filings = pd.concat((filings, pd.DataFrame(r_tmp.json())), axis = 0)
        time.sleep(.1)
    filings.reset_index(drop = True, inplace = True)
    return filings


def get_10_form_filing(cik: str, user_mail: str, accession_number: str, primary_doc: str, min_length_extract_from_body: int, raw_html: bool = False) -> str:

    """
    This function returns the 10 K or Q form filing of a company.

    Parameters:
    -----------
    cik: cik identifier used by the SEC
    accession_number: identifier used for submissions at the SEC
    primary_doc: second identifier
    user_mail: the mail which is used as a header information which is demanded by the SEC
    min_length_extract_from_body: Some of the SEC sites contain information in paragraph tags, others don't. If a text has less than this value, the site does 
    probably not contain its text content in paragraph texts. If this condition is met, we try to get the content of tags other than the paragraph tag.
    raw_html: if set to true, returns the raw html in str format

    """

    request_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{primary_doc}"
    headers = {"User-Agent": f"{user_mail}"}
    r = requests.get(request_url, headers = headers)
    time.sleep(.1)
    if raw_html:
        text = r.text
    else:
        sel = Selector(r)
        text = " ".join(sel.xpath("/html/body//p//text()").extract())
        text = re.sub("\s+", " ", text)
        if len(text) == 0 or len(text) < min_length_extract_from_body:
            text = " ".join(sel.xpath("/html/body//text()").extract())
            text = re.sub("\s+", " ", text)

    return text


def get_raw_submission(cik: str, user_mail: str, accession_number: str) -> str:

    """
    This function returns a text file which contains all information for a filing submission.

    Parameters:
    -----------
    cik: cik identifier used by the SEC
    user_mail: the mail which is used as a header information which is demanded by the SEC
    accession_number: accession number as provided by the get_filing_information method
    """

    headers = {"User-Agent": f"{user_mail}"}
    request_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{accession_number}.txt"
    res = requests.get(request_url, headers = headers)
    time.sleep(.1)
    return res.text


def find_documents(raw_text: str) -> dict:

    """
    This function identifies the parts which are included in the raw text file submission provided by the get_raw_submission method.

    Parameters:
    -----------
    raw_text: string which contains all parts of a filing submission
    """

    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_text)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_text)]

    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_text)]

    documents = {}
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        documents[doc_type] = raw_text[doc_start:doc_end]

    return documents


def extract_text_from_html(html_text: str, min_length_extract_from_body: int) -> str:

    """
    This is a helper function to extract text content from html strings.

    Parameters:
    -----------
    html_text: A html string
    min_length_extract_from_body: Some of the SEC sites contain information in paragraph tags, others don't. If a text has less than this value, the site does 
        probably not contain its text content in paragraph texts. If this condition is met, we try to get the content of tags other than the paragraph tag.
    """

    sel = Selector(text = html_text)
    text = " ".join(sel.xpath("/html/body//p//text()").extract())
    text = re.sub("\s+", " ", text)
    if len(text) == 0 or len(text) < min_length_extract_from_body:
        text = " ".join(sel.xpath("/html/body//text()").extract())
        text = re.sub("\s+", " ", text)
    return text


def get_8K_filing_and_exhibits(cik: str, user_mail: str, accession_number: str, min_length_extract_from_body: int, raw_html: bool = False, separator_character: str = "-;-", return_keys = False) -> str:

    """
    This function extracts 8K reports and exhibits which start with EX-99.

    Parameters:
    -----------
    cik: cik identifier used by the SEC
    user_mail: the mail which is used as a header information which is demanded by the SEC
    accession_number: accession number as provided by the get_filing_information method
    separator_character: the 8K filing and exhibits are separated by this symbol

    """

    raw_submission = get_raw_submission(cik, user_mail, accession_number)
    docs = find_documents(raw_submission)

    form_8k_string = ""
    form_8k_keys = ""
    for key in docs.keys():
        if re.match("8-K", key) or re.match("EX-", key): 
            form_8k_string += key + ":"
            if raw_html:
                form_8k_string += docs[key]
            else:
                form_8k_string += extract_text_from_html(docs[key], min_length_extract_from_body = min_length_extract_from_body)
            form_8k_string += separator_character
            form_8k_keys += key
            form_8k_keys += separator_character
    if return_keys:
        return form_8k_keys, form_8k_string
    else:
        return form_8k_string

