"""
Multiplist Sampling Filter Patterns v2

Based on research from:
- Kuhn, Cheney, & Weinstock (2000) - Epistemic stance framework
- Nussbaum, Sinatra, & Poliquin (2008) - Behavioral markers of stances
- Wu, Lee, Chai, & Tsai (2025) - Justification patterns

Key insight from Nussbaum et al.: Multiplists show "tolerance for inconsistencies"
and treat opposing views as equally valid WITHOUT weighing them. They "tended to 
just repeat or agree" rather than engage critically.

Strategy: Match multiplist patterns AND exclude evaluativist patterns to reduce
false positives from "polite evaluativists" who defer AFTER weighing.
"""

import re

# =============================================================================
# MULTIPLIST INCLUSION PATTERNS
# =============================================================================

# Strong multiplist indicators - high confidence signals
MULTIPLIST_STRONG_PATTERNS = [
    # ----- Core: Refusing to weigh/evaluate -----
    r'\bonly you can (decide|know|answer|figure|determine|say)\b',
    r'\bthat\'s (really )?(only )?(for you|yours) to decide\b',
    r'\bonly you (really )?know\b',
    r'\byou\'re the only one who (can|knows?|could)\b',
    r'\bno one (else )?(can|could|should) (tell|decide|know|say)\b',
    r'\bnobody can (tell|decide|answer) (that|this) (for|but) you\b',
    
    # ----- Core: Equal validity of all views -----
    r'\b(both|all) (views?|perspectives?|sides?|opinions?) (are )?(equally )?(valid|legitimate|right)\b',
    r'\bneither (is|are) (more )?(right|wrong|correct|better)\b',
    r'\beveryone\'s entitled to (their|an) (own )?(opinion|view|perspective)\b',
    r'\bwho\'s to say (what\'s|which is|if)\b',
    r'\bwho am i to (judge|say|tell you)\b',
    r'\bnot (for me|my place) to (say|judge|decide|tell)\b',
    
    # ----- Core: Knowledge as purely subjective -----
    r'\bit\'s (really )?(all |totally |completely )?subjective\b',
    r'\bthere (is|\'s) no (right|wrong|correct|objective|single|one) answer\b',
    r'\bno (right|wrong) (answer|choice|decision) (here|in this)\b',
    r'\bthere\'s no way to (know|say|tell) (what\'s|which is) (right|better|correct)\b',
    
    # ----- Inconsistency tolerance (Nussbaum) -----
    # Presenting contradictory options as equally valid
    r'\b(some say|some think|some people).{1,40}(others say|others think|other people).{1,40}(both|all).{1,20}(valid|right|true|sense)\b',
    r'\byou could (stay|leave|go|do).{1,30}(or|but).{1,30}(stay|leave|go|do).{1,30}(either|both).{1,20}(fine|ok|valid|right)\b',
    
    # ----- Pure relativism -----
    r'\beveryone\'s (situation|relationship|circumstance|experience) is (so )?(different|unique)\b',
    r'\bwhat (works|is right|is best) for (one person|someone|others) (might|may|won\'t|doesn\'t)\b',
    r'\bdifferent (things|approaches|choices) work for different (people|folks|individuals)\b',
    r'\bit\'s (a |so )?(personal|individual) (thing|matter|choice|decision)\b',
    
    # ----- Explicit refusal to judge -----
    r'\bi (can\'t|cannot|won\'t|couldn\'t) (tell you what to|say what you should|judge)\b',
    r'\bi\'m not (going to|gonna|here to) (tell you|judge|say) (what|how)\b',
    r'\bi (don\'t|can\'t) (think|feel) (it\'s )?(my|anyone\'s) place to\b',
]

# Moderate multiplist indicators - need additional context or multiple matches
MULTIPLIST_MODERATE_PATTERNS = [
    # ----- Deferring to individual (weaker forms) -----
    r'\bthat\'s (just |totally |really )?(your|a personal) (call|decision|choice)\b',
    r'\b(it\'s |that\'s )?(up to you|your call)\b',
    r'\byou (have to|need to|gotta|should) (decide|figure.+out) (for yourself|on your own)\b',
    r'\bdo what(ever)? (feels|seems|is) right (to|for) you\b',
    r'\bwhatever (you think|works for you|feels right|you decide)\b',
    r'\bgo with (your gut|what feels right|your instinct)\b',
    r'\btrust (yourself|your gut|your instincts)\b',
    
    # ----- Weak subjectivity markers -----
    r'\bit (really )?depends on (the person|who you are|you as a person|your values)\b',
    r'\bdepends on (the person|who you ask)\b',
    r'\beveryone is (so )?different\b',
    r'\bpeople (are|have) different (views|opinions|values|needs)\b',
    
    # ----- Hedging without substance -----
    r'\bjust my (opinion|two cents|perspective|view|thoughts?)\b',
    r'\bthat\'s just (how i|what i) (see it|think|feel)\b',
    r'\bi\'m just (one person|some random|a stranger)\b',
    
    # ----- Non-engagement (Nussbaum: "tended to agree or repeat") -----
    r'\bi (totally )?(agree|hear you|understand|get it)\s*[.!]?\s*$',  # Agreement at end without adding
    r'\bthat (totally )?makes sense\s*[.!]?\s*$',  # Validation without engaging
    r'\bi (can )?see (that|why|how)\s*[.!]?\s*$',  # Acknowledgment without contribution
]


# =============================================================================
# EVALUATIVIST EXCLUSION PATTERNS
# =============================================================================

# If these patterns appear, the sample is likely evaluativist, not multiplist
# Use these to EXCLUDE samples that match multiplist patterns but show weighing
EVALUATIVIST_EXCLUSION_PATTERNS = [
    # ----- Weighing/comparing language -----
    r'\b(stronger|weaker|better|worse|more likely|less likely) (argument|case|option|choice|reason)\b',
    r'\bon balance\b',
    r'\bweighing (the|your|these)\b',
    r'\b(the |one )?(bigger|larger|main|real|core|key) (issue|concern|problem|question) (is|here|seems)\b',
    r'\bthe (strongest|weakest|best|most compelling) (argument|point|reason|case)\b',
    
    # ----- Making reasoned recommendations -----
    r'\bi\'d (lean toward|suggest|recommend|argue|say)\b',
    r'\bi (would )?(lean toward|lean towards|tend to think|tend to say)\b',
    r'\bmy (recommendation|suggestion|take|assessment) (is|would be)\b',
    r'\bif i (were|was) (you|in your (shoes|position|situation))\b.{1,50}(i\'d|i would|i might)\b',
    
    # ----- Evidence-based reasoning -----
    r'\bbased on (what you\'ve|what you) (said|described|shared|told|written)\b',
    r'\bfrom what (you\'ve |you )(described|said|shared|told)\b',
    r'\bgiven (what you|the circumstances|the situation|that)\b',
    r'\bthe (fact|evidence|pattern) (that|suggests|indicates)\b',
    
    # ----- Engaging with tradeoffs -----
    r'\bon (the )?one hand\b.{1,100}\bon the other (hand)?\b',
    r'\bthe (tradeoff|trade-off|downside|upside|pro|con) (is|here|being)\b',
    r'\b(that said|however|but|although)\b.{1,50}\b(i\'d|i would|i think|i still)\b',
    
    # ----- Conditional/calibrated reasoning -----
    r'\bif (he|she|they|it) (is|are|has|have|shows?|seems?)\b.{1,50}\bthen\b',
    r'\b(it )?depends on (whether|if|how much|what)\b',  # Conditional depends (vs. relativistic)
    r'\bassuming (that|he|she|they|you)\b',
    
    # ----- Comparative evaluation -----
    r'\b(more|less) (concerning|worrying|promising|encouraging|important)\b',
    r'\b(this|that) (seems|is|feels) (more|less) (like|important|serious)\b',
    r'\bthe (more|most|less|least) (important|concerning|relevant|significant)\b',
    
    # ----- Epistemic humility WITH position -----
    # These show the person HAS a position but acknowledges uncertainty
    r'\bi could be wrong,? but (i think|i\'d|i believe|my sense)\b',
    r'\bi (think|believe|feel)\b.{1,50}\b(but|though|although) (i\'m|i am) not (sure|certain)\b',
    r'\bmy (current|initial|tentative) (view|thinking|take|sense) is\b',
]


# =============================================================================
# ABSOLUTIST EXCLUSION PATTERNS  
# =============================================================================

# If these patterns appear strongly, sample might be absolutist not multiplist
ABSOLUTIST_EXCLUSION_PATTERNS = [
    # ----- Certainty/definitiveness -----
    r'\byou (need|have|must|should) to (leave|stay|go|dump|end|break)\b',
    r'\b(this is|that\'s) (a )?(huge |major |big |serious )?(red flag|dealbreaker|deal breaker)\b',
    r'\bthere\'s no (excuse|reason|justification) for\b',
    r'\b(never|always) (put up with|accept|tolerate|allow)\b',
    r'\bno (one|person) should (ever |have to )?(put up|deal|stay|tolerate)\b',
    
    # ----- Dismissing other views -----
    r'\banyone who (thinks|says|believes)\b.{1,30}\b(wrong|crazy|naive|foolish)\b',
    r'\bthere\'s (only )?one (answer|option|choice|thing to do)\b',
    r'\b(obviously|clearly|undeniably|undoubtedly|without question)\b',
    r'\bit\'s (obvious|clear) (that|what)\b',
    
    # ----- Appeals to authority as final word -----
    r'\b(studies|research|experts?|science) (show|prove|say|agree)\b.{0,30}[.!]',  # Ending with authority appeal
    r'\bit\'s (a )?(fact|proven|established) that\b',
    
    # ----- Strong directives -----
    r'\b(dump|leave|run|get out|end it)( him| her| them)?[.!]+\s*$',  # Imperative at end
    r'\bdo not (stay|put up|tolerate|accept)\b',
    r'\byou (absolutely|definitely|clearly) (need|have|should|must) to\b',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compile_patterns(pattern_list):
    """Compile a list of regex patterns for efficiency."""
    return [re.compile(p, re.IGNORECASE) for p in pattern_list]


def count_matches(text, compiled_patterns):
    """Count total matches across all patterns."""
    count = 0
    matched_patterns = []
    for pattern in compiled_patterns:
        matches = pattern.findall(text)
        if matches:
            count += len(matches)
            matched_patterns.extend(matches)
    return count, matched_patterns


def is_likely_multiplist_candidate(text, 
                                    min_strong_matches=1,
                                    min_moderate_matches=2,
                                    max_evaluativist_matches=1,
                                    max_absolutist_matches=0):
    """
    Determine if a text sample is a good multiplist candidate for labeling.
    
    Returns:
        tuple: (is_candidate, score, details)
    """
    # Compile patterns (in production, do this once at module load)
    strong = compile_patterns(MULTIPLIST_STRONG_PATTERNS)
    moderate = compile_patterns(MULTIPLIST_MODERATE_PATTERNS)
    eval_excl = compile_patterns(EVALUATIVIST_EXCLUSION_PATTERNS)
    abs_excl = compile_patterns(ABSOLUTIST_EXCLUSION_PATTERNS)
    
    # Count matches
    strong_count, strong_matches = count_matches(text, strong)
    moderate_count, moderate_matches = count_matches(text, moderate)
    eval_count, eval_matches = count_matches(text, eval_excl)
    abs_count, abs_matches = count_matches(text, abs_excl)
    
    # Decision logic
    has_strong = strong_count >= min_strong_matches
    has_moderate = moderate_count >= min_moderate_matches
    too_evaluativist = eval_count > max_evaluativist_matches
    too_absolutist = abs_count > max_absolutist_matches
    
    # A candidate needs multiplist signals AND not too many exclusion signals
    is_candidate = (has_strong or has_moderate) and not too_evaluativist and not too_absolutist
    
    # Score for ranking candidates (higher = more likely multiplist)
    # score = (strong_count * 3) + (moderate_count * 1) - (eval_count * 2) - (abs_count * 2)
    
    # details = {
    #     'strong_matches': strong_matches,
    #     'strong_count': strong_count,
    #     'moderate_matches': moderate_matches,
    #     'moderate_count': moderate_count,
    #     'evaluativist_matches': eval_matches,
    #     'evaluativist_count': eval_count,
    #     'absolutist_matches': abs_matches,
    #     'absolutist_count': abs_count,
    #     'score': score,
    # }
    
    return is_candidate #, score, details


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Should be MULTIPLIST candidate (true multiplist)
        (
            "Honestly, only you can decide what's right here. Everyone's situation is different. "
            "Some people would leave, others would stay. There's no right answer - it's totally subjective. "
            "Who am I to judge what you should do?",
            True,
            "True multiplist - refuses to weigh, all views equal"
        ),
        
        # Should NOT be candidate (evaluativist with deference)
        (
            "Based on what you've described, the pattern of broken promises is concerning. "
            "I'd lean toward leaving because trust is foundational. That said, couples therapy "
            "could help if he's genuinely committed to change. Ultimately you know details I don't, "
            "but I think leaving is the stronger option here.",
            False,
            "Evaluativist - weighs options, makes recommendation, then defers"
        ),
        
        # Should NOT be candidate (absolutist)
        (
            "You need to leave him. This is a huge red flag. Anyone who does this is manipulative "
            "and there's no excuse for that behavior. Don't let him convince you otherwise. "
            "Get out now.",
            False,
            "Absolutist - certainty, directives, dismisses alternatives"
        ),
        
        # Should be MULTIPLIST candidate (deceptive multiplist - harder case)
        (
            "Some people think you should leave, others think staying and working on it is valid. "
            "Both perspectives make sense honestly. It's really a personal decision that only you "
            "can make. Different things work for different people.",
            True,
            "Multiplist - presents both as equal, doesn't weigh"
        ),
        
        # Edge case - should NOT be candidate (evaluativist with hedging)
        (
            "Just my two cents, but I think the bigger issue here is the pattern of behavior. "
            "On one hand, people can change. On the other hand, he's had chances. "
            "I'd lean toward protecting yourself, though I could be wrong.",
            False,
            "Evaluativist - hedges but still weighs and recommends"
        ),
    ]
    
    print("Testing multiplist filter patterns...\n")
    print("=" * 80)
    
    for text, expected, description in test_cases:
        is_candidate, score, details = is_likely_multiplist_candidate(text)
        status = "✓" if is_candidate == expected else "✗"
        
        print(f"\n{status} {description}")
        print(f"   Expected: {'candidate' if expected else 'NOT candidate'}")
        print(f"   Got: {'candidate' if is_candidate else 'NOT candidate'} (score: {score})")
        print(f"   Strong matches ({details['strong_count']}): {details['strong_matches'][:3]}...")
        print(f"   Moderate matches ({details['moderate_count']}): {details['moderate_matches'][:3]}...")
        print(f"   Evaluativist exclusions ({details['evaluativist_count']}): {details['evaluativist_matches'][:3]}...")
        print(f"   Absolutist exclusions ({details['absolutist_count']}): {details['absolutist_matches'][:3]}...")
    
    print("\n" + "=" * 80)
    print("Testing complete.")
