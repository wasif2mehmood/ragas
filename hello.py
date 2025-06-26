import pandas as pd
from datasets import Dataset

# Manual data creation with all required fields
manual_data = {
    'question': [
        "How can a Data Integration Specialist utilize Telegram for data integration using document loaders?",
        "Wut iz the purpuse of the RedditPostsLoader in data integrashun?",
        "How can a Data Integration Specialist utilize the Facebook ChatLoader for data integration?",
        "How can I load data from Reddit for integration purposes?",
        "how do you load tweets from twitter using document loaders?",
        "What BeautifulSoup do?",
        "What is the function of Firecrawl in the context of document loaders for web pages?",
        "How does BeautifulSoup assist in loading and parsing HTML web pages?",
        "Wut iz the purpoze of a Sitemap in the context of web page document loaders?",
        "Wut is the purpse of PyPDF in loading PDF documnts?"
    ],
    'ground_truth': [
        "A Data Integration Specialist can utilize Telegram for data integration by using the TelegramChatFileLoader, which is a document loader designed to load data from Telegram messaging platforms.",
        "The RedditPostsLoader is used to load documents from the Reddit social media platform, facilitating data integration from this source.",
        "A Data Integration Specialist can utilize the Facebook ChatLoader to load data from Facebook Chat, which is an instant messaging app and platform, for seamless integration and analysis.",
        "You can use the RedditPostsLoader to load documents from Reddit for integration.",
        "To load tweets from Twitter, you can use the TwitterTweetLoader, which is a document loader designed for this purpose.",
        "BeautifulSoup is used to load and parse HTML web pages.",
        "Firecrawl is an API service that can be deployed locally, and its hosted version offers free credits for loading and parsing web pages.",
        "BeautifulSoup is used in conjunction with urllib to load and parse HTML web pages.",
        "A Sitemap is used to scrape all pages on a given sitemap, allowing for comprehensive loading and parsing of web pages.",
        "PyPDF is used to load and parse PDFs as a document loader package."
    ],
    'answer': [
        "To utilize Telegram for data integration, use the TelegramChatFileLoader from the messaging services document loaders.",
        "The RedditPostsLoader serves to extract and load posts from Reddit's social platform for data integration workflows.",
        "The Facebook ChatLoader enables loading chat data from Facebook Messenger for integration into data pipelines.",
        "Use the RedditPostsLoader from social platform document loaders to extract Reddit data.",
        "Use the TwitterTweetLoader to extract tweet data from Twitter for your integration needs.",
        "BeautifulSoup is a Python library used for parsing HTML and XML documents from web pages.",
        "Firecrawl is an API service for web scraping that can be deployed locally or used as a hosted service with free credits.",
        "BeautifulSoup works with urllib to fetch and parse HTML content from web pages into structured data.",
        "Sitemap document loaders use XML sitemaps to systematically scrape all pages listed in a website's sitemap.",
        "PyPDF uses the pypdf library to extract text and metadata from PDF files for document processing."
    ],
    'contexts': [
        ["Social Platforms: Document loaders for social media platforms include TwitterTweetLoader, RedditPostsLoader. Messaging Services: TelegramChatFileLoader, WhatsAppChatLoader, DiscordChatLoader, FacebookChatLoader, MastodonTootsLoader."],
        ["Social Platforms: The RedditPostsLoader allows loading documents from Reddit social media platform along with TwitterTweetLoader for Twitter integration."],
        ["Messaging Services document loaders include TelegramChatFileLoader, WhatsAppChatLoader, DiscordChatLoader, FacebookChatLoader for loading data from messaging platforms."],
        ["Social platform document loaders include TwitterTweetLoader for Twitter and RedditPostsLoader for Reddit data integration."],
        ["Social Platforms document loaders: TwitterTweetLoader for loading tweets, RedditPostsLoader for Reddit posts integration."],
        ["Webpages loaders: Web uses urllib and BeautifulSoup to load and parse HTML web pages as a package-based solution."],
        ["Webpages document loaders include: Web (urllib + BeautifulSoup), Unstructured, RecursiveURL, Sitemap, Firecrawl API service with free credits."],
        ["Web document loader uses urllib and BeautifulSoup to load and parse HTML web pages from various web sources."],
        ["Webpage loaders include Sitemap for scraping all pages on a given sitemap, RecursiveURL for child links, and other web scraping tools."],
        ["PDF document loaders: PyPDF uses pypdf to load and parse PDFs, Unstructured for open source PDF loading, Amazon Textract for AWS API."]
    ],
    'reference_contexts': [
        ["Social Platforms and Messaging Services document loaders allow loading from various platforms including Telegram, Reddit, Twitter, Facebook, WhatsApp, Discord, and Mastodon."],
        ["Social Platforms document loaders include TwitterTweetLoader and RedditPostsLoader for loading documents from social media platforms."],
        ["Messaging Services document loaders include TelegramChatFileLoader, WhatsAppChatLoader, DiscordChatLoader, FacebookChatLoader, and MastodonTootsLoader."],
        ["Social Platforms provide document loaders for Twitter and Reddit among other social media platforms for data integration."],
        ["Social Platforms document loaders allow loading documents from Twitter using TwitterTweetLoader and Reddit using RedditPostsLoader."],
        ["Webpages section describes document loaders for loading web pages using various tools including urllib and BeautifulSoup."],
        ["Webpages document loaders include various tools like Web, Unstructured, RecursiveURL, Sitemap, Firecrawl, Docling, Hyperbrowser, and AgentQL."],
        ["Web document loader utilizes urllib and BeautifulSoup as packages for loading and parsing HTML web pages."],
        ["Webpages document loaders provide multiple options including Sitemap for scraping sitemap pages and other web scraping utilities."],
        ["PDFs section covers document loaders for PDF files including PyPDF, Unstructured, Amazon Textract, MathPix, and other PDF processing tools."]
    ],
    'synthesizer_name': ['single_hop_specifc_query_synthesizer'] * 10
}

# Create DataFrame
bulk_df = pd.DataFrame(manual_data)
# Save to CSV
bulk_df.to_csv('ragas_dataset.csv', index=False)
print("Saved to ragas_dataset.csv")
# Convert to HuggingFace Dataset
syn_dataset = Dataset.from_dict(manual_data)

print("Dataset created successfully!")
print(f"Features: {syn_dataset.features}")
print(f"Number of samples: {len(syn_dataset)}")