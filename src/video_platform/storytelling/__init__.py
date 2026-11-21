"""Indian story adaptation subsystem (model-agnostic).

Pipeline order (each step sees the output of the previous one):
  1. DialogueContextAnalyzer      understand every scene: who, where, mood, intentions, relationships
  2. IndianNameGenerator +        character identity registry (names, forms of address,
     RegionalLanguageProfileManager   per-character speaking profiles)
  3. ScreenplayEnhancementEngine + RegionalEnvironmentAdapter   what each scene may enhance; which real
                                                                places fit (from the location database)
  4. IndianDialogueRewriter +     write the Indian screenplay for the scene (not a translation),
     NaturalConversationGenerator    with natural conversation and an audio direction per line
  5. DialogueDurationOptimizer +  timing, names, slang, geography, originality, naturalness, screenplay
     ScreenplayQualityEvaluator +    scores; low-scoring lines are rewritten (version history kept)
     CreativeReferenceManager
  6. EmotionAndDeliveryPlanner    turns each line's direction into TTS parameters, pauses and gain

Modules build prompts and validate answers; the stages (stages/adaptation.py) call the models.
"""

from .context import DialogueContextAnalyzer
from .delivery import EmotionAndDeliveryPlanner
from .duration import DialogueDurationOptimizer
from .enhancement import ScreenplayEnhancementEngine
from .environment import LocationDatabase, RegionalEnvironmentAdapter
from .models import ModelSelector
from .names import IndianNameGenerator
from .profiles import RegionalLanguageProfileManager
from .quality import ScreenplayQualityEvaluator
from .references import CreativeReferenceManager
from .rewriter import IndianDialogueRewriter, NaturalConversationGenerator

__all__ = [
    "CreativeReferenceManager", "DialogueContextAnalyzer", "DialogueDurationOptimizer", "EmotionAndDeliveryPlanner",
    "IndianDialogueRewriter", "IndianNameGenerator", "LocationDatabase", "ModelSelector",
    "NaturalConversationGenerator", "RegionalEnvironmentAdapter", "RegionalLanguageProfileManager",
    "ScreenplayEnhancementEngine", "ScreenplayQualityEvaluator",
]
