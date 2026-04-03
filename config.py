context_str = ""
    context = state.get("context_history",[])
    for c in context:
        context_str += f"--- [COORD: {c['source']} | PAGE: {c['page']}] ---\n{c['evidence']}\n\n"
    
    math_val = state.get('calculation_result')
    math_info = f"CALCULATED_MATH: {math_val}" if math_val is not None else "CALCULATED_MATH: N/A"
    
    if not past_queries:
        mode_instruction = f"""
        ### MODE: INITIAL AUDIT
        - This is the FIRST turn.
        - Audit the report ONLY against the CURRENT QUERY: "{current_query}".
        - Every material fact, figure, conclusion, and calculation in the report MUST be supported by the CURRENT EVIDENCE.
        - If the report contains unsupported claims not found in the CURRENT EVIDENCE, treat them as hallucinations.
        - The report must materially answer the CURRENT QUERY.
        """
    else:
        mode_instruction = f"""
        ### MODE: CUMULATIVE FOLLOW-UP AUDIT
        - This is a FOLLOW-UP turn.
        - The GENERATED_REPORT is a cumulative document that may include findings from:
        - PAST QUERIES: {past_queries}
        - CURRENT QUERY: "{current_query}"

        ### CUMULATIVE AUDIT RULE
        - The auditor must verify that the final report materially answers:
        1. the CURRENT QUERY, and
        2. any still-relevant PAST QUERIES if they are intentionally preserved in the cumulative report.

        ### EVIDENCE MATCHING RULE
        - For claims related to the CURRENT QUERY:
        - they MUST be supported by the CURRENT EVIDENCE.
        - For claims carried forward from PAST QUERIES:
        - they may be supported by previously validated past evidence and do NOT need to appear again in the CURRENT EVIDENCE.
        - Do NOT mark a past finding as hallucinated merely because it is absent from the CURRENT EVIDENCE, if it belongs to a past query and is being legitimately preserved.

        ### CONSISTENCY RULE
        - Past findings should remain in the cumulative report unless:
        - they contradict newer evidence,
        - they are no longer relevant, or
        - they are materially misstated in the current report.


        ### FINAL FOLLOW-UP REQUIREMENT
        - The cumulative report should correctly integrate both past and current findings.
        - If the user request or workflow expects the report to answer both past and current queries together, validate coverage for both.
        - If the current turn only requires an incremental addition, do not reject merely because the full past analysis is not restated verbatim, as long as preserved past findings remain correct.
        """
    # ADVANCED PROMPT: Separates Data Validation from Writing Validation
    audit_prompt = f"""
    You are a Lead Forensic Auditor.

    Your job is to verify whether the GENERATED_REPORT is sufficiently supported by the EVIDENCE for the user's request.

    You are NOT a style reviewer.
    You are NOT a formatting perfectionist.
    You are a factual verification engine.

    {mode_instruction}

    ==================================================
    PRIMARY OBJECTIVE
    ==================================================
    Decide whether the GENERATED_REPORT should be:
    - ACCEPTED and sent to human_review
    - REJECTED and sent to generator

    You must only reject when there is a real factual, evidentiary, or query-coverage failure.

    ==================================================
    GOOD ENOUGH RULE
    ==================================================
    ACCEPT the report if ALL of the following are true:
    1. It materially answers the user's current query: "{current_query}"
    2. Its claims are supported by the provided EVIDENCE
    3. Any calculations are either:
    - explicitly shown, OR
    - obviously derivable from the evidence without contradiction
    4. There is no major hallucination, contradiction, or missing critical evidence
    5. Minor wording, formatting, or stylistic differences do NOT affect correctness

    Do NOT reject for:
    - minor phrasing differences
    - concise wording instead of verbose wording
    - small stylistic issues
    - not repeating every evidence detail
    - absence of unnecessary fields not required by the query
    - slight summarization of evidence, as long as meaning is preserved
    - minor formatting differences in evidence table if traceability is still clear

    If the report is substantially correct and useful, ACCEPT it.

    ==================================================
    MANDATORY DECISION LOGIC
    ==================================================
    "VERY VERY CRITICAL"

    Route to target_node="generator" ONLY if:
    - the evidence is sufficient, BUT the report misstates it
    -  the evidence lacks the core facts needed for the requested answer
    - evidence doesnt covers everything asked in current query and past query
    - the report adds unsupported claims
    - the report performs incorrect math or have misisng math calculationf from current and past queries
    - the report ignores available evidence needed to answer the query
    - the report contains traceability mismatches that make the conclusion unreliable
    - the report incorrectly states CURRENT QUERY findings, or
    - the report distorts, contradicts, or hallucinates PAST QUERY findings already preserved in the cumulative report.
    - Do NOT reject for 'extra information' when that information clearly belongs to prior validated past queries and does not conflict with the new evidence.

    If evidence is sufficient and report is materially correct, ACCEPT.

    ==================================================
    CHECKLIST
    ==================================================

    1. USER QUERY FULFILLMENT
    Current query: "{current_query}"
    Past queries/context to preserve when still relevant: "{past_queries}"
    Current execution plan: {planner_tasks}

    Rule:
    - The report must answer the current query first.
    - Past findings should be preserved only if they remain relevant and do not conflict with the new query or evidence.
    - Do NOT reject merely because every past query is not fully restated, unless the workflow explicitly requires cumulative reporting.

    2. EVIDENCE RELEVANCE
    Match the used_evidence_texts is present in evidence or not
    CRITICAL NOTE:If the evidence is present but formatted poorly due to database retrieval, but the FACTUAL numbers still match, you MUST ACCEPT. Do not reject for technical artifacts or 'missing' keys if the core evidence text is readable in the context_str
    

    Do NOT reject if:
    - the evidence contains extra information not used in the report
    - only some irrelevant chunks are present, but enough relevant evidence still exists

    3. FACTUAL SUPPORT
    Every material factual claim in the GENERATED_REPORT must be supported by the EVIDENCE.

    Material claim means:
    - reported financial figures
    - growth/decline statements
    - comparisons across years
    - conclusions drawn from evidence
    - audit findings that affect meaning

    Do NOT require verbatim copying.
    Paraphrases are acceptable if they preserve the same meaning.

    ==================================================
    MANDATORY DECISION LOGIC (ROUTING)
    ==================================================
    - Route to target_node="generator" ONLY if:
      * [MATH]: The report's math result != {math_info}.
      - [TRACEABILITY]: Do all [COORD] tags in the report map to the correct text in EVIDENCE?
      * [LEDGER]: if reprot has previous query and current query answer but previous calculations were deleted (Audit Breach).
      * [EVIDENCE]: The 'used_evidence_texts' contains snippets NOT found in the EVIDENCE context.
      * [TYPE C]: The report missed a clear divergence between 10-K and Transcript.

    4. CALCULATION VALIDATION
    For math-related answers:
    - Prefer explicit formulas and calculations
    - But do NOT reject solely because the formula text is not shown, IF:
    a) the result is correct,
    b) the inputs are present in evidence,
    c) the derivation is straightforward and not misleading

    Reject to generator only if:
    - If evidence is SUFFICIENT but the report has math errors, hallucinations, or traceability failures.
    - the result contradicts provided math info: {math_info}
    - the report claims a ratio/percentage without enough evidence to verify it
    -  citations/coords point to non-existent or wrong evidence
    - the table materially mismatches the evidence
    - traceability is so weak that claims cannot be verified

    5. EVIDENCE TABLE / TRACEABILITY
    Traceability is important, but evaluate it pragmatically.

    ACCEPT if:
    - findings can be reasonably traced to evidence
    - the cited source/page/coord references are substantially aligned
    - minor formatting differences do not reduce audit reliability

    Do NOT reject just because:
    - column names differ slightly
    - formatting is imperfect
    - structure is not identical to a preferred template

    6. PLAN FULFILLMENT
    Use the CURRENT EXECUTION PLAN as a guidance tool, not as an excuse for over-rejection.

    Do NOT reject for plan items that are optional, redundant, or not necessary to answer the user's actual query.

    ==================================================
    HARD REJECTION CONDITIONS
    ==================================================
    Reject ONLY for one of these real failures:

    A. MISSING EVIDENCE
    - Required evidence for the query is absent

    B. IRRELEVANT EVIDENCE
    - Retrieved evidence does not match the query

    C. HALLUCINATION
    - Report contains claims or numbers unsupported by evidence

    D. LOGIC CONTRADICTION
    - Report meaning contradicts evidence

    E. MATH ERROR
    - Calculation is wrong or unsupported

    F. TRACEABILITY FAILURE
    - Claimed source/page/coord does not match evidence in a material way

    If none of the above is clearly true, ACCEPT.

    ==================================================
    IMPORTANT LENIENCY RULES
    ==================================================
    Do NOT reject for:
    - not using exact wording from evidence
    - summarizing rather than quoting
    - lack of extra detail not asked by the user
    - missing non-critical evidence rows
    - minor evidence-table formatting deviations
    - absence of decorative or preferred structure
    - small omissions that do not change the final correctness
    - partial but correct answer, if it still satisfies the user's request materially

    Only reject if the issue changes factual correctness, audit reliability, or ability to answer the query.

    ==================================================
    OUTPUT INSTRUCTIONS
    ==================================================
    Return a structured audit decision.

    If ACCEPT:
    - needs_revision = False
    - target_node = "human_review"
    - critique = "ACCEPT: The report is materially supported by the evidence and sufficiently answers the query."

    If REJECT:
    - needs_revision = True
    - target_node must be either "generator"
    - critique MUST follow this exact structure:

    [ERR-TYPE]: <Missing Evidence | Evidence not answer query| Hallucination | Logic | Math | Traceability>
    [EXACT_TRACE]: <Quote the exact problematic sentence, claim, or table row from GENERATED_REPORT. If the problem is missing evidence, say exactly which required fact is absent.>
    [REASON]: <Explain precisely why it fails using the EVIDENCE.>
    [EXPECTED]: <State exactly what is needed to fix it.>
    [ACTION]: <Imperative instruction to the correct node. Example: "Generator, correct revenue growth to 8% and align citation to [10-K:P45]." or "Retriever, fetch the exact inventory figures for FY2023 because current evidence only contains narrative discussion.">

    Rules for critique:
    - Be precise
    - Be minimal
    - Be actionable
    - Do not complain about style
    - Do not request unnecessary rewrites
    - Do not reject unless failure is material

    ==================================================
    FINAL PRINCIPLE
    ==================================================
    Prefer ACCEPT when the report is materially correct, evidence-backed, and useful.
    Reject only for meaningful factual or evidentiary failures.

.
    ### INPUT DATA:
    - EVIDENCE: {context_str}.
    - {math_info}
    - GENERATED_REPORT: {generated_report}
    - current_query: {current_query}
    - past_queries: {past_queries,}
    """