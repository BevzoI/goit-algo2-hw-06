import logging
from colorama import init, Fore
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup  # NEW

# ==========================
# Logger & colorama setup
# ==========================
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================
# Stop words
# ==========================
STOP_WORDS = {
    "the", "and", "is", "in", "to", "a", "of", "it", "that", "on", "was", "with",
    "as", "for", "at", "by", "an", "be", "this", "have", "from", "or", "but", "not"
}


class WordCountMapReduce:
    def __init__(self, url: str, chunk_size: int = 2000, max_workers: int = 4, top_n: int = 10):
        self.url = url
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.top_n = top_n
        self.words = []
        self.word_counts = {}

    def fetch_text(self) -> str:
        logger.info(Fore.BLUE + f"Fetching text from: {self.url}")
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")  # NEW
            return soup.get_text()  # Extract only visible text
        except requests.RequestException as e:
            logger.error(Fore.RED + f"Failed to fetch from URL. Error: {e}")
            return ""

    def text_clean_and_split(self, text: str) -> None:
        logger.info(Fore.GREEN + "Cleaning and splitting text into words...")
        cleaned_text = re.sub(r"[^a-zA-Z]+", " ", text).lower()
        self.words = [word for word in cleaned_text.split() if word not in STOP_WORDS]  # NEW
        logger.info(Fore.GREEN + f"Total words extracted (filtered): {len(self.words)}")

    @staticmethod
    def mapper(chunk: list) -> dict:
        local_count = defaultdict(int)
        for word in chunk:
            local_count[word] += 1
        return local_count

    @staticmethod
    def reducer(dict_list: list) -> dict:
        final_dict = defaultdict(int)
        for d in dict_list:
            for word, cnt in d.items():
                final_dict[word] += cnt
        return final_dict

    def parallel_mapreduce(self) -> None:
        logger.info(Fore.CYAN + "Starting parallel MapReduce...")
        chunks = [self.words[i: i + self.chunk_size] for i in range(0, len(self.words), self.chunk_size)]

        logger.info(Fore.YELLOW + f"Submitting {len(chunks)} chunks to mapper with {self.max_workers} workers.")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.mapper, chunk) for chunk in chunks]
            partial_dicts = [f.result() for f in futures]

        logger.info(Fore.YELLOW + "Mapper phase completed. Now reducing...")
        self.word_counts = self.reducer(partial_dicts)
        logger.info(Fore.GREEN + f"Reduce phase completed. Unique words found: {len(self.word_counts)}")

    def visualize_top_words(self, save_to_file: bool = False, filename: str = "top_words.png") -> None:
        if not self.word_counts:
            logger.warning(Fore.RED + "No word counts to visualize. Possibly the text was empty.")
            return

        sorted_items = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_items[:self.top_n]

        words = [item[0] for item in top_words]
        counts = [item[1] for item in top_words]

        plt.figure(figsize=(8, 5))
        plt.barh(words, counts, color="skyblue")
        plt.title(f"Top {self.top_n} Most Frequent Words")
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_to_file:
            plt.savefig(filename)  # NEW
            logger.info(Fore.GREEN + f"Chart saved to file: {filename}")
        else:
            plt.show()

    def run(self, save_to_file: bool = False) -> None:
        logger.info(Fore.CYAN + "=== Starting WordCountMapReduce process ===")
        text = self.fetch_text()
        if not text:
            logger.warning(Fore.RED + "No text retrieved. Exiting run().")
            return

        self.text_clean_and_split(text)
        self.parallel_mapreduce()
        self.visualize_top_words(save_to_file=save_to_file)


def main():
    url = "https://gutenberg.net.au/ebooks01/0100021.txt"
    mapreduce = WordCountMapReduce(url=url, chunk_size=2000, max_workers=4, top_n=10)
    mapreduce.run(save_to_file=True)  # Save chart to PNG


if __name__ == "__main__":
    main()
