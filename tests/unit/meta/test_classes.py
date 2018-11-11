import pandas as pd
import pytest

from epic_kitchens.meta import classes

classes._verb_classes = pd.DataFrame(
    {
        "verb_id": [0, 1],
        "class_key": ["take", "put"],
        "verbs": [["take", "pick-up"], ["put", "put-down", "put-onto"]],
    }
)


classes._noun_classes = pd.DataFrame(
    {
        "noun_id": [0, 1, 2],
        "class_key": ["Nothing", "pan", "pan:dust"],
        "nouns": [
            ["Nothing"],
            ["pan", "pan:sauce", "saucepan"],
            ["pan:dust", "dustpan"],
        ],
    }
)


@pytest.mark.parametrize("verb_class,verb", [(0, "take"), (1, "put"), (1, "pan:sauce")])
def test_class_to_verb(verb_class, verb):
    assert classes.class_to_verb(verb_class) == verb


@pytest.mark.parametrize(
    "verb_class,verb", [(0, "take"), (0, "pick-up"), (1, "put"), (1, "put-down")]
)
def test_class_to_verb(verb_class, verb):
    assert classes.verb_to_class(verb) == verb_class


@pytest.mark.parametrize(
    "noun_class,noun", [(0, "Nothing"), (1, "pan"), (1, "pan:sauce"), (2, "pan:dust")]
)
def test_nouns_to_class(noun_class, noun):
    assert classes.noun_to_class(noun) == noun_class


@pytest.mark.parametrize(
    "noun_class,noun", [(0, "Nothing"), (1, "pan"), (2, "pan:dust")]
)
def test_class_to_nouns(noun_class, noun):
    assert classes.class_to_noun(noun_class) == noun
