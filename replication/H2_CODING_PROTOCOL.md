# H2 Narrative Coding Human Validation Protocol

## Overview
This protocol guides human coders in validating the rule-based algorithmic coding of LLM-generated rationales for H2 (frame-outcome association). The gold standard sample consists of 200 rationales randomly selected from the high-burden condition (~10% of the sample).

---

## Task Description

For each rationale, you will independently code whether it contains evidence of **five theoretically derived frames**:

1. **Access Barriers** — practical obstacles to vaccination
2. **Coercive Backlash** — resistance to mandates/penalties  
3. **Collective Responsibility** — civic duty, protecting others
4. **Autonomy Infringement** — personal choice, freedom concerns
5. **Procedural Legitimacy** — fairness, transparency, trust

*Note: Frames are not mutually exclusive. A single rationale may contain 0, 1, or multiple frames.*

---

## Coding Instructions

### General Principles

1. **Code what is explicitly stated or clearly implied.** Do not infer frames that are not supported by the text.
2. **Focus on the reasoning provided**, not your own judgment about the scenario.
3. **Binary coding:** For each frame, enter `1` if present, `0` if absent.
4. **Work independently.** Do not consult the algorithmic labels or other coders during coding.

---

## Frame Definitions with Examples

### 1. Access Barriers
**Definition:** The rationale attributes reduced willingness to practical obstacles in obtaining vaccination: cost, time requirements, work schedule inflexibility, childcare needs, travel distance, documentation burden, or other logistical constraints.

**Keywords/concepts:**
- Distance, travel time, transportation
- Wait time, queuing, scheduling
- Work conflicts, inflexible schedule
- Childcare, family obligations
- Cost, affordability, income constraints
- Documentation, paperwork, registration requirements

**Example (POSITIVE):**
> "Multiple barriers significantly reduce willingness: inflexible work schedule makes taking time off difficult, required in-person registration with documentation gathering adds administrative burden, and 1-2 hour wait time is substantial."

**Example (NEGATIVE):**
> "She has concerns about vaccine side effects and wants to wait for more safety data before making a decision." *(No mention of access/logistical barriers)*

---

### 2. Coercive Backlash
**Definition:** The rationale describes resistance, resentment, distrust, or reduced legitimacy arising specifically from perceived mandates, penalties, enforcement, or coercive policy design—distinct from general distrust or personal choice concerns.

**Keywords/concepts:**
- Mandate, mandatory, compulsory, required
- Penalty, fine, punishment, consequence
- Enforcement, imposed, forced
- Backlash, resistance, resentment (when linked to coercion)
- Loss of trust specifically due to mandates

**Example (POSITIVE):**
> "The requirement for immediate vaccination within 5 days, backed by penalties for non-compliance, creates resentment and distrust in the policy."

**Example (NEGATIVE):**
> "She doesn't trust the government health authorities in general." *(General distrust, not backlash against mandates)*

**Boundary case:** General distrust of authorities without reference to mandates/penalties → **Code as 0** for coercive_backlash, may be autonomy_infringement if framed as choice violation.

---

### 3. Collective Responsibility
**Definition:** The rationale frames vaccination as a collective or civic responsibility emphasizing protection of others, community benefit, public health outcomes, or social duty—distinct from individual health benefits.

**Keywords/concepts:**
- Protecting others, community, herd immunity
- Public health, population health
- Civic duty, responsibility to society
- Contagion, transmission to vulnerable people
- Collective benefit, common good

**Example (POSITIVE):**
> "Vaccination helps protect vulnerable community members and contributes to herd immunity, which is a shared social responsibility."

**Example (NEGATIVE):**
> "She wants to protect her own health." *(Individual benefit, not collective responsibility)*

**Important:** Simple mentions of "public health" without framing as collective duty or community protection → **Code as 0** (insufficient evidence).

---

### 4. Autonomy Infringement
**Definition:** The rationale emphasizes personal choice, individual rights, bodily autonomy, or freedom from external interference—framed as concerns about personal decision-making rather than collective obligations or practical barriers.

**Keywords/concepts:**
- Personal choice, individual decision
- Bodily autonomy, right to refuse
- Freedom, liberty, rights
- Personal preference, individual circumstances
- Control over own health decisions

**Example (POSITIVE):**
> "She believes vaccination should be a personal choice and resents external pressure to make this medical decision."

**Example (NEGATIVE):**
> "She doesn't have time to go to the clinic." *(Access barrier, not autonomy infringement)*

---

### 5. Procedural Legitimacy
**Definition:** The rationale references procedural fairness, transparency, trust in institutions, legitimacy of the decision-making process, or equitable treatment—distinct from trust in medical/scientific claims.

**Keywords/concepts:**
- Fairness, equitable treatment
- Transparency, clear communication
- Trust in authorities (when procedural)
- Legitimacy of process
- Due process, proper procedure

**Example (POSITIVE):**
> "The policy was communicated transparently and implemented fairly, which increases trust in the health authorities' decision-making process."

**Example (NEGATIVE):**
> "She trusts that the vaccine is safe and effective." *(Trust in medical claims, not procedural legitimacy)*

---

## Coding Procedure

### Step 1: Set up
1. Open `h2_gold_coding_sheet.csv` in Excel, Google Sheets, or similar
2. You will see columns: `rationale_id`, `rationale_text`, and five `human_*` columns
3. Enter your initials in the coder_id field (or as column header suffix if multiple coders)

### Step 2: Code each rationale
For each row (rationale):
1. Read the `rationale_text` completely
2. For each of the five frames:
   - Ask: "Does this rationale explicitly or clearly implicitly contain this frame?"
   - Enter `1` if YES, `0` if NO
   - If uncertain, use the definitions and examples above to resolve
3. Move to next rationale

### Step 3: Quality check
Before submitting:
1. Review 5-10 random rows to verify consistency
2. Check that you haven't left any cells blank (all should be 0 or 1)
3. Save your completed file as `h2_gold_coding_completed_[your_initials].csv`

---

## Blinding Protocol

**CRITICAL:** This coding must be BLINDED to the algorithmic labels.

- Do NOT consult `h2_jev_coding.csv` or any algorithmic output
- Do NOT view the Pattern Dictionary until AFTER completing your coding
- If you are the same person who developed the algorithm, complete the human coding BEFORE reviewing algorithm outputs
- Focus on the text itself and your independent judgment

---

## Inter-Coder Reliability (if multiple coders)

If two independent coders are available:
1. Each coder completes the full 200-rationale sample independently
2. Use `score_h2_gold_coding.py` to calculate:
   - Per-frame agreement percentage
   - Cohen's κ (kappa) for each frame
   - Precision, Recall, F1 relative to consensus
3. Resolve discrepancies through discussion for final gold standard

---

## Validation Metrics

After human coding is complete, the following metrics will be reported:

| Metric | Description |
|:---|:---|
| **Exact Agreement** | % of rationales where human = algorithm |
| **Precision** | Of algorithm positives, % that are true positives |
| **Recall** | Of human positives, % caught by algorithm |
| **F1** | Harmonic mean of precision and recall |
| **Cohen's κ** | Agreement corrected for chance (κ > 0.6 acceptable, > 0.8 good) |

---

## Expected Time

- 200 rationales × 1-2 minutes each = **3.5-6 hours total**
- Can be completed in multiple sessions
- Recommended: Complete in 2-3 sessions to maintain consistency

---

## Contact

Questions about frame definitions or coding decisions: [Your contact]

---

**Version:** 1.0  
**Date:** October 2026  
**Protocol for:** Value-Sensitivity LLM Audit (H2 Validation)
