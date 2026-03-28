import json
from typing import Dict, Any

import pandas as pd
import stringcase
from torch.distributions.utils import lazy_property

from utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET,
    AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA,
)
from utils.objaverse_utils import get_objaverse_annotations


class VIDAObject(dict):
    ALWAYS_KEYS = {"isObjaverse", "synset", "lemma", "objaverseType", "naturalType"}
    CLUSTER_KEYS = {}
    AFFORDANCE_KEYS = {"affordances"}

    def __init__(self, thor_obj: Dict[str, Any], affordances=None, clusters=None):
        super().__init__()
        self._thor_obj = thor_obj
        self._cache = {}
        self._affordances = affordances
        self._clusters = clusters

    def _valid_affordances(self):
        assert self._affordances is not None, "pre-computed affordances required"
        cat = self["synset"]
        return pd.DataFrame.from_records(
            [
                dict(cluster=entry["cluster"], cat=cat, use=entry["use"])
                for entry in self._affordances
                if cat in entry["cats"]
            ]
        )

    def set_affordances_and_clusters(self, affordances, clusters):
        self._affordances = affordances
        self._clusters = clusters

    @lazy_property
    def is_objaverse(self):
        return self._thor_obj["assetId"] in get_objaverse_annotations()

    def __getitem__(self, item):
        if self.is_objaverse and item == "objectType" and self._thor_obj[item] == "Undefined":
            return self._thor_obj["objectId"].split("|")[0]

        if item in self._thor_obj:
            return self._thor_obj[item]

        if item in self._cache:
            return self._cache[item]

        asset_id = self._thor_obj["assetId"]

        if item == "isObjaverse":
            return self.is_objaverse

        elif item == "synset":
            if self.is_objaverse:
                self._cache[item] = get_objaverse_annotations()[asset_id]["synset"]
            else:
                self._cache[item] = AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET[
                    self._thor_obj["objectType"]
                ]

        elif item == "lemma":
            if self.is_objaverse:
                self._cache[item] = get_objaverse_annotations()[asset_id]["most_specific_lemma"]
            else:
                self._cache[item] = AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA[
                    self._thor_obj["objectType"]
                ]

        elif item == "objaverseType":
            ## TODO Temp fix for debugging replace with synset - prefer to deprecate entirely
            if self.is_objaverse:
                self._cache[item] = get_objaverse_annotations()[asset_id]["synset"]
            else:
                self._cache[item] = AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET[
                    self._thor_obj["objectType"]
                ]

        elif item == "naturalType":
            # TODO this should be done much better, (if we really want to use it)!
            self._cache[item] = stringcase.snakecase(
                self["objaverseType"].replace("Obja", "")
            ).replace("_", " ")

        elif item == "affordances":
            self._cache[item] = self._valid_affordances()

        elif self.is_objaverse and item in get_objaverse_annotations()[asset_id]:
            self._cache[item] = get_objaverse_annotations()[asset_id][item]

        elif not self.is_objaverse and item == "description":
            self._cache[item] = f"undescribed THOR item, type {self._thor_obj['objectType']}"

        else:
            raise ValueError(f"Unknown key {item}")

        return self._cache[item]

    def __setitem__(self, key, value):
        if key in self._thor_obj:
            self._thor_obj[key] = value
        else:
            self._cache[key] = value

    def _key_set(self):
        keys = set(self._thor_obj.keys())
        keys.update(self._cache.keys())
        keys.update(self.ALWAYS_KEYS)

        if self._clusters is not None:
            keys.update(self.CLUSTER_KEYS)

        if self._affordances is not None:
            keys.update(self.AFFORDANCE_KEYS)

        if self.is_objaverse:
            keys.update(get_objaverse_annotations()[self._thor_obj["assetId"]].keys())

        return keys

    def keys(self):
        return iter(self._key_set())

    def values(self):
        return map(self.__getitem__, self.keys())

    def __iter__(self):
        for key in self.keys():
            yield key

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __contains__(self, key):
        return (
            key in self._thor_obj
            or key in self._cache
            or key in self.ALWAYS_KEYS
            or (self.is_objaverse and key in get_objaverse_annotations())
            or (self._clusters is not None and key in self.CLUSTER_KEYS)
            or (self._affordances is not None and key in self.AFFORDANCE_KEYS)
        )

    def __eq__(self, other):
        if not isinstance(other, VIDAObject):
            return False
        return self._thor_obj == other._thor_obj and self._cache == other._cache

    def __str__(self):
        return json.dumps({**self._thor_obj, **self._cache}, indent=2)

    # noinspection PyStatementEffect
    def __repr__(self):
        return (
            f"VIDAObject("
            f"objectId={self['objectId']},"
            f" objectType={self['objectType']},"
            f" assetId={self['assetId']},"
            f" isObjaverse={self['isObjaverse']},"
            f")"
        )
