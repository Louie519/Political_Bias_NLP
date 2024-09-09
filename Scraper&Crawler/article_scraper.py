import json
import re
import numpy as np
from fundus import PublisherCollection, Crawler, RSSFeed, NewsMap, Sitemap, Requires
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from fundus.scraping.filter import inverse, regex_filter, lor, land




#publisher_names_right = ['Fox News', 'The Gateway Pundit', 'The Washington Free Beacon']
#publisher_names_center = ['CNBC', 'Reuters', 'Voice Of America', 'Wired']
#publisher_names_left = ['The Intercept', 'The Nation', 'The New Yorker']




# Function to filter publishers based on names
def filter_publishers(publishers, names):
    return [publisher for publisher in publishers if publisher.name in names]

def main():
    publisher_names_right = ['The Gateway Pundit', 'The Washington Free Beacon']
    publisher_names_center = ['CNBC', 'Reuters', 'Wired']
    publisher_names_left = ['The Intercept', 'The New Yorker']

    # Get the filtered list of publishers
    filtered_publishers_right = filter_publishers(PublisherCollection.us, publisher_names_right)
    filtered_publishers_center = filter_publishers(PublisherCollection.us, publisher_names_center)
    filtered_publishers_left = filter_publishers(PublisherCollection.us, publisher_names_left)

    source_types = [RSSFeed, NewsMap, Sitemap]

    # Initialize the crawlers with the desired settings
    crawler_right = Crawler(*filtered_publishers_right, restrict_sources_to=source_types)
    crawler_center = Crawler(*filtered_publishers_center, restrict_sources_to=source_types)
    crawler_left = Crawler(*filtered_publishers_left, restrict_sources_to=source_types)

    bias_mapping = {
        crawler_right: "Right",
        crawler_center: "Center",
        crawler_left: "Left"
    }

    articles_list = []
    for crawl in [crawler_left, crawler_center, crawler_right]:
        crawler = crawl
        for article in crawler.crawl(max_articles=1, url_filter=inverse(regex_filter("politic"))):
            try:
                # Extract the source URL
                source_url = article.html.requested_url
                
                # Extract the publisher from the source URL
                match = re.search(r'www\.(.*?)\.com', source_url)
                publisher = match.group(1) if match else "Unknown"
                
                # Convert the article to a dictionary
                article_data = article.to_json("title", "plaintext", "publishing_date")
                
                # Add the source URL and publisher to the dictionary
                article_data["source"] = source_url
                article_data["publisher"] = publisher
                article_data["bias"] = bias_mapping.get(crawler, "Unknown")
                
                # Append the article dictionary to the list
                articles_list.append(article_data)
            except KeyError as e:
                print(f"KeyError: {e} for article: {article}")
            except Exception as e:
                print(f"Error: {e} for article: {article}")

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(articles_list)
    df.to_csv('articles_7/31.csv', index=False)

if __name__ == "__main__":
    main()


