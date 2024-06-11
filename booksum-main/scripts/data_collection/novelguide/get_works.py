"""
/*
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 */
"""

from builtins import zip, str, range
import pdb, os, csv, re, io, string
import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
from tqdm import tqdm
from shutil import rmtree
from nltk.tokenize import word_tokenize, sent_tokenize

# PARAMS
MAIN_SITE = 'https://web.archive.org/web/20210225014436/https://www.novelguide.com/'
SEED_URL = 'https://web.archive.org/web/20210225014436/https://www.novelguide.com/title/'

alphabet_list = string.ascii_lowercase + '1'

errors_file = open("link_errors.txt","w")

def scrape_index_pages(seed_page):
# For each summary info

    scraped_links = []

    for char in alphabet_list:
        page_no = 1
        books_page = seed_page + char

        while(True):

            try:

                soup = BeautifulSoup(urllib.request.urlopen(books_page), "html.parser")
                items = soup.findAll("ul", {"class": "search-title"})
                books = items[0].findAll("li")

                # # Go over each section
                for index, item in enumerate(books):
                    # Parse section to get bullet point text

                    item_title = item.find("a").text
                    item_url = item.find("a").get("href")

                    print ("item_title: ", item_title.strip())
                    print ("item_url: ", item_url.strip())
                    print ("\n")

                    scraped_links.append({
                        "title": item_title.strip(),
                        "url": urllib.parse.urljoin(MAIN_SITE, item_url.strip())
                    })

            except Exception as e:
                print (books_page, str(e))
                errors_file.write(books_page + "\t" + str(e) + "\n")

                break

            books_page = seed_page + char + "?page=" + str(page_no)
            page_no += 1
            
    return scraped_links

# generate literature links
scraped_data = scrape_index_pages(SEED_URL)

with open("literature_links.tsv", "w") as fd:
    for data in scraped_data:
        fd.write("%s\t%s\n" % (data["title"], data["url"]))
