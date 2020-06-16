import scrapy
from scrapy.crawler import CrawlerProcess

# don't import - just copy to notebook or script and define the urls you would like. 

url_short = ['https://assets.datacamp.com/production/repositories/2560/datasets/19a0a26daa8d9db1d920b5d5607c19d6d8094b3b/all_short']

class WebSpider(scrapy.Spider):
    name = 'template_spider'
    # start_requests method - required by scrapy
    def start_requests( self ):
        # URLS GO HERE \/
        urls = url_short
        for url in urls:
            yield scrapy.Request(url = url, callback=self.parse)
            
    # Use css locator or xpath to get information. 
    def parse(self, response):
        crs_titles = response.xpath('//h4[contains(@class,"block__title")]/text()').extract()
        crs_descrs = response.xpath('//p[contains(@class,"block__description")]/text()').extract()
        for crs_title, crs_descr in zip( crs_titles, crs_descrs ):
            dc_dict[crs_title] = crs_descr
    
    # use the response.follow() method to navigate to new pages.
    # add second layer parser method to gather  material from new pages. 
    
    
# Initialize the dictionary **outside** of the Spider class
# dc_dict = dict()

# Run the Spider
# process = CrawlerProcess()
# process.crawl(WebSpider)
# process.start()
# to print the output do something simple like:
# for k in dc_dict:
#     print(dc_dict[v])