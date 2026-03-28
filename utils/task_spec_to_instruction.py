import math
import random
from typing import Any, Dict
from functools import lru_cache

import prior
from nltk.corpus import wordnet2022 as wn
from utils.constants.object_constants import AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET

try:
    from phonemizer.backend import EspeakBackend
except:
    print("WARNING: phonemizer is not available")

try:
    import inflect
except:
    print("WARNING: inflect is not available")


from utils.constants.template_verbs import GOTO_VERBS
from utils.objaverse_utils import get_objaverse_annotations, OBJAVERSE_HOME_COMMIT_ID
from utils.synsets.hypernyms import get_all_synsets_in_vida, generate_all_hypernyms_with_exclusions

PHYSICAL_ENTITY_SYNSET = wn.synset("physical_entity.n.01")

SYNSET_TO_BEST_LEMMA = prior.load_dataset(
    "objaverse-plus",
    which_dataset="synset_to_best_lemma",
    revision=OBJAVERSE_HOME_COMMIT_ID,
)["train"].data


_CACHED_LEMMAS = None

VOWEL_PHONEMES = "aeiouɐʌəæɑɛɪɔʊ"
_phonemize_backend = None
_inflect_backend = None

_inflect_error_shown = False
_phonemizer_error_shown = False


def phonemize_backend():
    global _phonemize_backend
    if _phonemize_backend is None:
        try:
            _phonemize_backend = EspeakBackend("en-us")
        except:
            emsg = "WARNING: the espeak backend is not available"
            global _phonemizer_error_shown
            if not _phonemizer_error_shown:
                print(emsg)
                _phonemizer_error_shown = True
            raise RuntimeError(emsg)
    return _phonemize_backend


def inflect_backend():
    global _inflect_backend
    if _inflect_backend is None:
        try:
            _inflect_backend = inflect.engine()
        except:
            emsg = "WARNING: the inflect engine is not available"
            global _inflect_error_shown
            if not _inflect_error_shown:
                print(emsg)
                _inflect_error_shown = True
            raise RuntimeError(emsg)
    return _inflect_backend


def normalize(text):
    if ".n." in text:
        return (
            best_lemma(text, precomputed=True).lower().replace("_", " ").strip().strip(".;/,'\"\\")
        )
    else:
        return text.strip().lower().replace("_", " ").strip().strip(".;/,'\"\\")


def is_physical_entity(synset):
    if isinstance(synset, str):
        synset = wn.synset(synset)
    return PHYSICAL_ENTITY_SYNSET in synset.lowest_common_hypernyms(PHYSICAL_ENTITY_SYNSET)


def best_lemma_via_specificity(synset_str):
    synset = wn.synset(synset_str)
    cur_synset_is_physical_entity = is_physical_entity(synset)
    min_num_synsets = 100000
    best_lemma = None
    for ln in synset.lemma_names():
        if cur_synset_is_physical_entity:
            num_synsets = len([s for s in wn.synsets(ln, pos=wn.NOUN) if is_physical_entity(s)])
        else:
            num_synsets = len(wn.synsets(ln, pos=wn.NOUN))
        if num_synsets < min_num_synsets:
            min_num_synsets = num_synsets
            best_lemma = ln
    assert best_lemma is not None
    return best_lemma


def best_lemma(synset_str, precomputed=True):
    if precomputed:
        all_lemmas = load_synset_lemmas()["best"]
        return all_lemmas[synset_str]
    synset_str = wn.synset(synset_str).name()

    if synset_str in SYNSET_TO_BEST_LEMMA:
        best_lemma = SYNSET_TO_BEST_LEMMA[synset_str]
    else:
        best_lemma = best_lemma_via_specificity(synset_str)
    assert best_lemma is not None
    return best_lemma


def simple_lemma(synset_str, precomputed=True):
    if precomputed:
        all_lemmas = load_synset_lemmas()["simple"]
        return all_lemmas[synset_str]
    synset = wn.synset(synset_str)
    lemmas = synset.lemmas()
    if lemmas:
        return lemmas[0].name()
    return None


def load_synset_lemmas():
    global _CACHED_LEMMAS
    if _CACHED_LEMMAS is None:
        all_hypernyms = set()
        for syn in list(AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.values()) + get_all_synsets_in_vida():
            all_hypernyms.update(
                generate_all_hypernyms_with_exclusions(syn, excluded=set())
            )  # includes self

        _CACHED_LEMMAS = {
            "simple": {
                syn.name(): simple_lemma(syn.name(), precomputed=False) for syn in all_hypernyms
            },
            "best": {
                syn.name(): best_lemma(syn.name(), precomputed=False) for syn in all_hypernyms
            },
        }

    return _CACHED_LEMMAS


@lru_cache(maxsize=None)
def find_det(text):
    try:
        phonemes = phonemize_backend().phonemize([text])[0]
        if phonemes[0] in VOWEL_PHONEMES:
            return "an"
        else:
            return "a"
    except RuntimeError:
        first_letter = text[0]
        if first_letter in "aeiou":
            return "an"
        else:
            return "a"


def choose_det(text):
    """e.g. `a basketball` or `an icecream`"""
    return f"{find_det(normalize(text).split()[0])} {text}"


def make_source_obj(task_params):
    """e.g. basketball"""
    if "synsets" not in task_params:
        if "target_object_type" not in task_params:
            task_params["target_object_type"] = task_params["object_types"][0]
        return normalize(task_params["target_object_type"])
    return normalize(task_params["synsets"][0])


def make_rel_attribute(task_params):
    """e.g. `chair furthest from the fridge` or `smallest vase`"""
    object_type = make_source_obj(task_params)
    rel_attr = task_params["rel_attribute"]
    if isinstance(rel_attr, tuple) or isinstance(rel_attr, list):
        from_to = "to" if normalize(rel_attr[0]) in ["closest"] else "from"
        return f"{normalize(object_type)} {normalize(rel_attr[0])} {from_to} the {normalize(rel_attr[1])}"
    else:
        return f"{normalize(rel_attr)} {normalize(object_type)}"


def make_local_ref(task_params):
    """e.g. `near a chair and a house plant` or `on a dining table`"""
    if task_params["reference_type"] == "near":
        ref = (
            f"near {choose_det(normalize(task_params['reference_synsets'][0]))}"
            f" and {choose_det(normalize(task_params['reference_synsets'][1]))}"
        )
    elif task_params["reference_type"] == "on":
        ref = f"on {choose_det(normalize(task_params['reference_synsets'][0]))}"
    else:
        raise NotImplementedError
    return normalize(ref)


def make_src_room(task_params):
    """e.g. in the bedroom"""
    return normalize(f"in the {normalize(task_params['room_type'])}")




def object_nav_type(task_params: Dict[str, Any]):
    """e.g. navigate to a basketball"""
    return normalize(f"{random.choice(GOTO_VERBS)} {choose_det(make_source_obj(task_params))}")



def clean_object_type(task_params: Dict[str, Any]):
    object_type = task_params["object_type"]
    if object_type.startswith("Obja"):
        object_type = object_type[len("Obja") :]
    return object_type






def clean_description(desc):
    desc = normalize(desc)

    # handler for `this is worn on your feet to allow support when you walk and run.`
    if desc.startswith("this is worn"):
        desc = "the thing that" + desc[len("this") :]

    to_remove_in_order = [
        "i think this might be",
        "it looks to be",
        "it looks like",
        "it looks almost like",
        "it looks more like",
        "it look like",
        "this looks like",
        "this appears to be",
        "it it",
        "it is",
        "it's",
        "it s",
        "this is",
        "these are",
        "it looks",
        "it could be",
        "it",
        "there is",
        "there are",
        "these look like",
        "actually",
        "here are",
    ]

    for prefix in to_remove_in_order:
        if desc.startswith(prefix):
            desc = normalize(desc[len(prefix) :])

    # `they` only appears as a typo in starting position?
    if desc.split()[0] in ["a", "an", "some", "they", "and", "am", "as", "sa", "various"]:
        desc = " ".join(["the"] + desc.split()[1:])

    # handler for `this connects to your computer and is used to type`
    if desc.startswith("this connects"):
        desc = "the thing that" + desc[len("this") :]

    # handler for a few remaining descriptions starting with `this`
    if desc.startswith("this"):
        desc = "the" + desc[len("this") :]

    # `He has long toes on his feet with a plaid backpack and plaid hat`
    if desc.startswith("he"):
        desc = "the man" + desc[len("he") :]

    if desc.startswith("i"):
        desc = "the" + desc[len("i") :]

    # `what remains of ...` doesn't need `the`, nor seeking for verbs
    if desc.split()[0] not in ["the", "what"]:
        desc = f"the {desc}"

    if desc.split()[0] == "the":
        words = desc.split()
        constraint_seen = False
        for it, word in enumerate(words):
            if word in ["is", "are", "shows", "contains", "has", "have"]:
                if not constraint_seen:
                    desc = " ".join(words[:it] + [f"that {word}"] + words[it + 1 :])
                break
            elif (
                word in ["that", "who", "with", "which", "where", "but", "and"]
                or "." in word
                or "," in word
                or ";" in word
            ):
                # Note: adding `and` breaks a handful of sentences like `go to the black and white book has stripes`,
                # but also handles some ungrammatical descriptions. Humans also make a lot of typos, anyway.
                constraint_seen = True
            elif word in ["that's", "thats"]:
                break

    desc = normalize(desc)

    # Remove trailing period
    if desc[-1] == ".":
        desc = desc[:-1]

    return desc


def object_nav_open_vocab(task_params: Dict[str, Any]):
    desc = clean_description(get_objaverse_annotations()[task_params["uid"]]["description"])
    return normalize(f"{random.choice(GOTO_VERBS)} {desc}")


@lru_cache(maxsize=None)
def pluralize(text):
    if text in [
        "assets",
        "athletics",
        "binoculars",
        "drygoods",
        "goggles",
        "remains",
        "dice",
        "eyeglasses",
        "shorts",
        "stairs",
        "starches",
        "sunglasses",
    ]:
        plural = text
    elif text == "lego":
        plural = "legos"
    elif text == "mine":
        plural = "mines"
    elif text == "thermos":
        plural = "thermoses"
    elif text == "pair of trousers":
        plural = "pairs of trousers"
    elif text in ["money", "paper money", "furniture", "office furniture", "bedroom furniture"]:
        plural = f"pieces of {text}"
    else:
        try:
            plural = inflect_backend().plural(text)
        except:
            plural = f"{text}s"
    return plural







REGISTERED_INSTRUCTION_TYPES = dict(
    EasyObjectNavType=object_nav_type,
    HardObjectNavType=object_nav_type,
    ObjectNavType=object_nav_type,
    
)


_ = load_synset_lemmas()


