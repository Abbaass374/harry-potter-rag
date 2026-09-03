"""Query router.

Classifies each incoming query into one of three routes *before* the expensive
RAG pipeline runs:

* ``greeting``      -> a canned friendly reply
* ``out_of_scope``  -> politely decline (don't hallucinate)
* ``hp_question``   -> run the full retrieve -> generate pipeline

The default classifier is a fast, deterministic keyword/heuristic one (great for
tests and offline use). An optional LLM-based fallback can be enabled for the
ambiguous cases.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Route labels
GREETING = "greeting"
OUT_OF_SCOPE = "out_of_scope"
HP_QUESTION = "hp_question"

# Canned responses used by the API for the non-RAG routes.
GREETING_RESPONSE = (
    "Hello! I'm a Harry Potter book assistant. Ask me anything about the seven "
    "books - characters, spells, places, or events - and I'll answer from the "
    "text itself and show you the sources."
)
OUT_OF_SCOPE_RESPONSE = (
    "I can only answer questions about the Harry Potter books. Try something "
    "like \"Who is Harry Potter's godfather?\" or \"What are the Deathly Hallows?\""
)

# Words that, on their own, signal a greeting / pleasantry.
_GREETING_WORDS = {
    "hi", "hello", "hey", "heya", "yo", "hiya", "greetings", "howdy",
    "thanks", "thank", "thankyou", "ty", "cheers",
    "bye", "goodbye", "morning", "afternoon", "evening", "night",
    "sup", "welcome",
}
# Low-signal filler tolerated inside an otherwise pure greeting.
_GREETING_FILLER = {
    "there", "you", "your", "the", "for", "a", "all", "so", "much", "very",
    "good", "day", "help", "u", "and", "im", "i", "am", "to", "me", "again",
    "please", "ok", "okay", "hows", "how", "are", "doing", "whats", "up",
}

# Harry-Potter-specific vocabulary. If any of these appears we treat the query
# as an in-scope HP question. Kept broad on purpose (characters, places, houses,
# creatures, spells, objects, concepts).
_HP_KEYWORDS = {
    # core
    "harry", "potter", "hogwarts", "voldemort", "dumbledore", "hermione",
    "granger", "ron", "weasley", "hagrid", "snape", "malfoy", "draco",
    "sirius", "dobby", "neville", "luna", "ginny", "mcgonagall", "hedwig",
    "bellatrix", "umbridge", "lupin", "pettigrew", "cho", "cedric", "fred",
    "george", "percy", "dursley", "dursleys", "riddle", "grindelwald",
    "moody", "kreacher", "wormtail", "voldemort's", "quirrell", "fawkes",
    "buckbeak", "peeves", "filch", "trelawney", "slughorn", "moaning",
    # places
    "gryffindor", "slytherin", "hufflepuff", "ravenclaw", "azkaban",
    "diagon", "hogsmeade", "gringotts", "privet", "godric's", "hallow",
    # concepts / objects / creatures
    "muggle", "muggles", "wizard", "witch", "wand", "wands", "spell", "spells",
    "horcrux", "horcruxes", "quidditch", "patronus", "dementor", "dementors",
    "phoenix", "basilisk", "hallows", "hallow", "sorting", "hat", "owl",
    "broomstick", "nimbus", "expelliarmus", "avada", "kedavra", "expecto",
    "patronum", "prophecy", "deathly", "goblet", "azkaban", "hallows",
    "chamber", "philosopher's", "sorcerer's", "godfather", "wizarding",
    "portkey", "polyjuice", "pensieve", "marauder", "occlumency",
}
# Multi-word phrases worth matching directly.
_HP_PHRASES = (
    "harry potter", "deathly hallows", "chamber of secrets", "goblet of fire",
    "order of the phoenix", "half-blood prince", "prisoner of azkaban",
    "philosopher's stone", "sorcerer's stone", "defense against the dark arts",
    "he who must not be named", "you-know-who", "sorting hat",
)


class Router:
    def __init__(self, generator=None, use_llm_fallback: bool = False):
        """
        Parameters
        ----------
        generator:
            Optional object exposing an Ollama ``_client`` (i.e. a ``Generator``)
            used for the LLM fallback classification of ambiguous queries.
        use_llm_fallback:
            If True and a generator is provided, queries that the heuristic marks
            ``out_of_scope`` get a second opinion from the LLM.
        """
        self.generator = generator
        self.use_llm_fallback = use_llm_fallback and generator is not None

    # -- public API ----------------------------------------------------------
    def classify(self, query: str) -> str:
        route = self._heuristic(query)
        if route == OUT_OF_SCOPE and self.use_llm_fallback:
            llm_route = self._llm_classify(query)
            if llm_route:
                route = llm_route
        logger.info("Routed query to '%s': %r", route, query[:80])
        return route

    # -- heuristic -----------------------------------------------------------
    def _heuristic(self, query: str) -> str:
        q = (query or "").strip().lower()
        if not q:
            return OUT_OF_SCOPE

        tokens = set(re.findall(r"[a-z']+", q))

        # 1) Pure greeting / thanks (only greeting words + a little filler).
        if tokens:
            leftovers = tokens - _GREETING_WORDS - _GREETING_FILLER
            has_greeting = bool(tokens & _GREETING_WORDS)
            if has_greeting and not leftovers:
                return GREETING

        # 2) Harry-Potter-specific vocabulary anywhere -> in scope.
        if tokens & _HP_KEYWORDS:
            return HP_QUESTION
        if any(phrase in q for phrase in _HP_PHRASES):
            return HP_QUESTION

        # 3) Otherwise, out of scope.
        return OUT_OF_SCOPE

    # -- optional LLM fallback ----------------------------------------------
    def _llm_classify(self, query: str) -> str | None:
        try:
            resp = self.generator._client.chat(
                model=self.generator.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the user's message into exactly one label: "
                            "'greeting', 'out_of_scope', or 'hp_question'. "
                            "'hp_question' means it is about the Harry Potter books. "
                            "Reply with only the label."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                options={"temperature": 0.0},
            )
            label = resp["message"]["content"].strip().lower()
            for candidate in (HP_QUESTION, GREETING, OUT_OF_SCOPE):
                if candidate in label:
                    return candidate
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM fallback classification failed: %s", exc)
        return None
