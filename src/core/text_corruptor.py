"""Script used to generate the original IMDB-C, as used for our paper."""

import collections
import dataclasses
import enum
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import string
import tempfile
import urllib
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import polyleven
from tqdm import tqdm

from datasets import load_dataset

DEFAULT_CACHE_DIR = "./.text_corruption_cache/"

MAX_COMMON_START_FOR_AUTOCOMPLETE = 5
MIN_COMMON_START_FOR_AUTOCOMPLETE = 3

THESAURUS_DOWNLOAD = (
    "https://raw.githubusercontent.com/zaibacu/thesaurus/master/en_thesaurus.jsonl"
)


def _levensthein_distance(word: str, words: List[str]) -> np.ndarray:
    """
    Calculates the Levenshtein distance between two words.
    Taken from (MIT licensed):
    https://github.com/nfmcclure/tensorflow_cookbook/blob/master/
        05_Nearest_Neighbor_Methods/03_Working_with_Text_Distances/03_text_distances.py
    :param word: the word to compare
    :param all_words: the list of all words
    :return: the Levenshtein distances to all words (including itself)
    """

    res = np.zeros(len(words))
    for i, w in enumerate(words):
        res[i] = polyleven.levenshtein(word, w)

    return res


def split_by_whitespace(strings: List[str]) -> List[List[str]]:
    """Splits a list of strings on whitespaces."""
    # using same regex as huggingface WhitespaceSplit
    # (see: https://huggingface.co/docs/tokenizers/python/latest/components.html)
    return [re.findall(r"\w+|[^\w\s]+", l) for l in strings]


def bad_autocompletes(
    word: str, start_bags: Dict[int, Dict[str, List[str]]], common_letters: int
) -> Optional[List[str]]:
    """Returns a list of words which start with the same letters as the passed word."""
    if common_letters < MIN_COMMON_START_FOR_AUTOCOMPLETE:
        # End of recursion, no common words found.
        # This will only rarely happen in sufficiently large datasets.
        return None

    common_letters = min(common_letters, len(word))
    start = word[:common_letters]

    try:
        bag = start_bags[common_letters][start]
    except KeyError:
        bag = []

    # Remove the word itself
    try:
        bag = [w for w in bag if w != word]
    except ValueError:
        pass

    # Handle no-match-found by checking fewer common letters
    if len(bag) == 0:
        # Gracefully handle case where no other words start with the selected number of same letters
        return bad_autocompletes(word, start_bags, common_letters=common_letters - 1)

    return bag


class CorruptionType(enum.Enum):
    """The four different corruption types, imitating natural corruptions."""

    TYPO = 0
    """Randomly replaced single chars. Imitates human typos."""
    SYNONYM = 1
    """Replacement with an (apparent) synonym. 
    
    As all context is ignored, this can (intentionally) lead to `wrong` replacements,
    given the content. This imitates word-by-word translations from a different language."""
    AUTOCOMPLETE = 2
    """Replacement with a word which starts with the same letters as the original word."""
    AUTOCORRECT = 3
    """Replacement with a very similar word.
    
    This imitates e.g. (failed) mobile phone input pattern recognitions."""


def _get_rng(seed):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    return rng


@dataclasses.dataclass
class CorruptionWeights:
    """Configuration of the weights of the different corruption types."""

    typo_weight: float = 0.05
    autocomplete_weight: float = 0.30
    autocorrect_weight: float = 0.30
    synonym_weight: float = 0.35


def _generate_corruption_types(
    seed: int,
    num_words: int,
    weights: CorruptionWeights,
) -> List[CorruptionType]:
    """A list of randomly chosen corruption types"""
    weights = np.array(
        [
            weights.typo_weight,
            weights.autocomplete_weight,
            weights.autocorrect_weight,
            weights.synonym_weight,
        ]
    )
    normalized_weights = weights / weights.sum()
    rng = _get_rng(seed)
    return [
        CorruptionType(rng.choice(4, p=normalized_weights)) for _ in range(num_words)
    ]


def _hash_text_to_int(words: List[str]) -> int:
    digest = _hash_text_to_str(words)
    # hex to int
    hashed = int(digest, 16)
    # some collisions are ok, and smaller hashes are nicer to look at, hence % 1kk
    return hashed % 1000000


def _hash_text_to_str(words: List[str]) -> str:
    return hashlib.md5(" ".join(words).encode("utf-8")).hexdigest()


class TextCorruptor(object):
    """A class for corrupting arbitrary english (not just imdb) text datasets."""

    def __init__(
        self,
        base_dataset: List[str],
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        dictionary_size: int = 4000,
        clear_cache: bool = False,
    ):
        if cache_dir is DEFAULT_CACHE_DIR:
            warnings.warn(
                "Using default cache directory, which is probably not what you want. "
                "Consider passing your own cache dir when creating a "
                "TextCorruptor instance. "
            )

        # Identifier of passed dataset
        self.base_ds_hash: str = _hash_text_to_str(
            base_dataset + [str(dictionary_size)]
        )
        # If cache dir is None: no caching
        self.cache_dir: Optional[str] = None
        if cache_dir is not None:
            self.cache_dir = os.path.join(cache_dir, self.base_ds_hash)
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            elif clear_cache:
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)

        self.common_words: List[str] = self._extract_common_words(
            base_dataset, dictionary_size
        )
        self.start_bags: Dict[int, Dict[str, List[str]]] = self._word_start_bags()
        self.lev_dist: np.ndarray = self._calculate_distances()
        self.thesaurus: Dict[str, List[str]] = self.load_bad_translations()

    def _extract_common_words(self, base_dataset: List[str], size: int) -> List[str]:
        """Identifies `size` most common words in a passed list of strings."""
        # Save chosen words using pickle
        if self.cache_dir is not None:
            words_file = os.path.join(self.cache_dir, "common-words.pkl")
            if os.path.exists(words_file):
                logging.info("Loading common words from cache")
                with open(words_file, "rb") as f:
                    return pickle.load(f)
        logging.info("Extracting common words")

        # Split on whitespaces
        logging.debug("[WORD EXTRACTION] Splitting dataset on whitespaces")
        words = split_by_whitespace(base_dataset)
        # Flatten samples and make lower case
        logging.debug("[WORD EXTRACTION] Flattening samples and making lower case")
        words = [w.lower() for l in words for w in l]
        # Remove words shorter than 4 characters
        logging.debug("[WORD EXTRACTION] Removing words shorter than 4 characters")
        words = [w for w in words if len(w) > 4]
        # Remove numbers
        logging.debug("[WORD EXTRACTION] Removing numbers")
        words = [w for w in words if not w.isdigit()]
        # Remove words which do not contain any letters
        logging.debug(
            "[WORD EXTRACTION] Removing words which do not contain any letters"
        )
        words = [w for w in words if any(c.isalpha() for c in w)]
        # Chose most frequent words
        logging.debug("[WORD EXTRACTION] Choosing most frequent words")
        chosen_words = dict(collections.Counter(words).most_common(size)).keys()
        chosen_words = list(chosen_words)
        # Sort alphabetically
        logging.debug("[WORD EXTRACTION] Sort chosen, unique words alphabetically")
        chosen_words.sort()

        logging.info("Finished extracting common words from imdb")

        if self.cache_dir is not None:
            with open(words_file, "wb") as f:
                pickle.dump(chosen_words, f)

        return chosen_words

    def _word_start_bags(self) -> Dict[int, Dict[str, List[str]]]:
        """Returns dictionaries of bags with equally starting words for different start sizes."""

        assert self.common_words is not None, "Common words not extracted yet."

        if self.cache_dir is not None:
            words_bags_file = os.path.join(self.cache_dir, "word-start-bags.pkl")
            if os.path.exists(words_bags_file):
                logging.info("Loading word-start-bags from cache")
                with open(words_bags_file, "rb") as f:
                    return pickle.load(f)

        def _group(num_start_chars: int) -> Dict[str, List[str]]:
            dict_res = dict()
            for word in self.common_words:
                if len(word) >= num_start_chars:
                    start = word[:num_start_chars]
                    if start not in dict_res:
                        dict_res[start] = []
                    dict_res[start].append(word)
            return dict_res

        with ThreadPoolExecutor() as executor:
            start_sizes = list(
                range(
                    MIN_COMMON_START_FOR_AUTOCOMPLETE,
                    MAX_COMMON_START_FOR_AUTOCOMPLETE + 1,
                )
            )
            dicts = list(executor.map(_group, start_sizes))

        result = {start_sizes[i]: dicts[i] for i in range(len(start_sizes))}

        if self.cache_dir is not None:
            with open(words_bags_file, "wb") as f:
                pickle.dump(result, f)

        return result

    def _calculate_distances(self) -> np.ndarray:
        """Calculates the Levenshtein distances between the passed chosen words."""
        if self.cache_dir is not None:
            distances_file = os.path.join(self.cache_dir, "distances.npy")
            if os.path.exists(distances_file):
                logging.info("Loading distances from cache")
                return np.load(distances_file)
        logging.info("Calculating levensthein distances")

        def _run_for_word(word):
            return _levensthein_distance(word, self.common_words)

        # Note: Runtime could further be improved by leveraging the symmetry in the distance matrix.
        #       At the moment, every entry is calculated twice.
        with ThreadPoolExecutor() as executor:
            distances = list(
                tqdm(
                    executor.map(_run_for_word, self.common_words),
                    total=len(self.common_words),
                    desc="Calculating Levenshtein distances",
                )
            )

        distances = np.array(distances, dtype=np.uint8)
        if self.cache_dir is not None:
            np.save(distances_file, distances)

        return distances

    def corrupt(
        self,
        texts: List[str],
        severity: float,
        seed: int,
        weights: Optional[CorruptionWeights] = None,
        force_recalculate: bool = False,
    ) -> List[str]:
        """Corrupts a dataset consisting of a list of strings (texts).

        The generation is set up such that:
        - With a given seed and severity, the same text will always be corrupted the same way
          independent of the order of the texts or the size of the passed texts list.
        - The same seed and severity will always result in the same corruption.
        - Higher severity will result in more corruptions, and include all (equivalent)
          corruptions which would have been applied to said sentence with lower severity.
          Hence, difficulty can be gradually increased using the severity parameter.
        - The severity loosly reflects the percentage of corrupted words in the output,
          for each entry in the texts list (considering only words with length>2).
          "Loosely" as in practice, sometimes corruptions fail (e.g. no similar word
          has been identified amongst the common words, or the word to be corrupted is not common)
          which will lead to fewer corrupted words.
        - The passed corruption weights are used to determine the probability of each corruption,
          but again only loosely as for some types of corruptions, if not possible,
          another type of corruption will be used instead.

        The documentation our github readme will provide a more detailed explanation
        of the different corruption types.

        Args:
            texts: List of strings to corrupt.
            severity: Probability of a character being replaced by a random character.
            seed: Seed to control randomness.
            weights: Weights for each corruption type.

        """

        assert 0 <= severity <= 1, "Severity must be between 0 and 1."

        if self.cache_dir is not None:
            ds_hash = _hash_text_to_str(texts)
            cache_file = os.path.join(
                self.cache_dir, "corrupted", f"{ds_hash}-{severity}-{seed}.pkl"
            )
            if os.path.exists(cache_file) and not force_recalculate:
                logging.info("Loading corrupted dataset from cache")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        if weights is None:
            weights = CorruptionWeights()

        assert 0 <= severity <= 1, "Severity must be between 0 and 1"

        def _corrupt_single_text(words: List[str]) -> str:
            """Corruption job for a single dataset entry (text)."""
            new_text = []

            # Seed which is independent of the order and number of texts in dataset
            sentence_seed = _hash_text_to_int(words) + seed

            # Seed handling to make sure seed for individual word is not influenced by the severity.
            #   Hence, higher severity will lead to the same, but more corruptions.
            #   We do this by choosing a corruption type for each word,
            #   and only then deciding which of these corruptions should be applied based on severity.
            corruption_types = _generate_corruption_types(
                sentence_seed, len(words), weights
            )
            corruption_indexes = np.arange(len(words))
            _get_rng(sentence_seed).shuffle(corruption_indexes)
            corruption_indexes = corruption_indexes[: round(len(words) * severity)]

            for i, word in enumerate(words):
                if np.sum(corruption_indexes == i) == 0 or len(word) < 2:
                    # Cases where no corruption should be applied
                    new_text.append(word)
                else:
                    # Cases where corruption should be applied
                    corruption = corruption_types[i]
                    word_seed = sentence_seed + i
                    corrupt_word = self._corrupt_word(word, word_seed, corruption)
                    new_text.append(corrupt_word)

            return " ".join(new_text)

        texts_as_words = split_by_whitespace(texts)
        corrupted_texts = []
        for i, text in tqdm(
            enumerate(texts_as_words),
            total=len(texts_as_words),
            desc="Corrupting dataset badges",
        ):
            corrupted_texts.append(_corrupt_single_text(text))

        if self.cache_dir is not None:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(corrupted_texts, f)

        return corrupted_texts

    def load_bad_translations(self) -> Dict[str, List[str]]:
        """Loads the bad translations. Downloads a wordnet thesaurus if needed.

        Override this method for datasets in non-english languages."""

        if self.cache_dir is not None:
            thesaurus_path = os.path.join(self.cache_dir, "bad_translations.pkl")
        else:
            random_str = _get_rng(None).choice(string.ascii_letters, size=10)
            thesaurus_path = os.path.join(
                tempfile.gettempdir(), f"bad_translations_{random_str}.pkl"
            )

        # Download file if not exists
        if not os.path.isfile(thesaurus_path):
            urllib.request.urlretrieve(THESAURUS_DOWNLOAD, thesaurus_path)

        # Load thesaurus
        with open(thesaurus_path) as f:
            data = [json.loads(line) for line in f]

        # Get simple bags of synonyms
        result = dict()
        for d in data:
            word = d["word"]
            synonyms = d["synonyms"]
            if len(synonyms) > 1:
                if word not in result:
                    result[word] = set()
                result.get(word).update(synonyms)

        for word in result.keys():
            result[word] = list(result.get(word))

        return result

    @staticmethod
    def _corrupt_typo(text, seed: int) -> str:
        letter_index = seed % len(text)
        candidate_letters = string.ascii_lowercase.replace(text[letter_index], "")
        # MD4 hash is faster than RNG and random enough
        random_candidate_index = _hash_text_to_int([text, str(seed)]) % len(
            candidate_letters
        )
        typo = candidate_letters[random_candidate_index]
        return text[:letter_index] + typo + text[letter_index + 1 :]

    def _corrupt_autocomplete(self, word, seed: int) -> str:
        candidates = bad_autocompletes(word, self.start_bags, common_letters=5)
        if candidates is None or len(candidates) == 0:
            # Edge case in small datasets, where no words start with the same letters as the word
            return self._corrupt_autocorrect(word, seed)
        # MD4 hash is faster than RNG and random enough
        random_candidate_index = _hash_text_to_int([word, str(seed)]) % len(candidates)
        return candidates[random_candidate_index]

    def _corrupt_autocorrect(self, word, seed: int) -> str:
        if word not in self.common_words:
            return word
        word_index = self.common_words.index(word)
        # Choose amongst the five most similar words, normalizing by the distance
        candidate_indices = np.argsort(self.lev_dist[word_index])[1:6]
        candidate_distances = 1 / self.lev_dist[word_index][candidate_indices]
        rng = _get_rng(seed)
        chosen_index = rng.choice(
            candidate_indices, p=candidate_distances / candidate_distances.sum()
        )
        return self.common_words[chosen_index]

    def _corrupt_synonym(self, word, seed: int) -> str:
        try:
            synonyms = self.thesaurus[word]
            if len(synonyms) == 0:
                raise KeyError
        except KeyError:
            # No synonyms found, so just return a typo instead
            return self._corrupt_typo(word, seed)

        # MD4 hash is faster than RNG and random enough
        method_salt = "_corrupt_synonym"
        random_candidate_index = _hash_text_to_int(
            [word, str(seed), method_salt]
        ) % len(synonyms)
        # Choose a synonym at random
        return synonyms[random_candidate_index]

    def _corrupt_word(self, w, seed, corruption_type: CorruptionType) -> str:
        if corruption_type == CorruptionType.TYPO:
            return self._corrupt_typo(w, seed)
        elif corruption_type == CorruptionType.AUTOCOMPLETE:
            return self._corrupt_autocomplete(w, seed)
        elif corruption_type == CorruptionType.AUTOCORRECT:
            return self._corrupt_autocorrect(w, seed)
        elif corruption_type == CorruptionType.SYNONYM:
            return self._corrupt_synonym(w, seed)
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nominal_train = load_dataset(
        "imdb", cache_dir="/expext2/deepgini/.external_datasets", split="train"
    )["text"]
    nominal_test = load_dataset(
        "imdb", cache_dir="/expext2/deepgini/.external_datasets", split="test"
    )["text"]
    corruptor = TextCorruptor(
        base_dataset=nominal_test + nominal_train, cache_dir=".imdb_test"
    )

    # Run only on a part of the dataset
    # nominal_test = nominal_test[:201]
    imdb_corrupted = corruptor.corrupt(nominal_test, severity=0.5, seed=1)
