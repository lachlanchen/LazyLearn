# Math Bank
## Core Equations
- [visible] None. No validated blackboard or slide equations survive for this lecture.
- [transcript-backed] \(\text{utility} = \text{pleasure} - \text{pain}\)
- [transcript-backed] \(\text{utility} = \text{happiness} - \text{suffering}\)
- [transcript-backed] \(\text{Right act} = \text{act that maximizes utility}\)
- [standard reconstruction] \(U(a) := H(a) - S(a)\), where \(H\) is total happiness and \(S\) is total suffering resulting from act \(a\)
- [standard reconstruction] \(a^\star = \arg\max_a U(a)\)
- [transcript-backed] Trolley driver case:
  \[
  a_{\text{turn}} : (N_{\text{die}}, N_{\text{live}}) = (1,5), \qquad
  a_{\text{straight}} : (N_{\text{die}}, N_{\text{live}}) = (5,1)
  \]
- [transcript-backed] Bridge case:
  \[
  a_{\text{push}} : (N_{\text{die}}, N_{\text{live}}) = (1,5), \qquad
  a_{\text{refrain}} : (N_{\text{die}}, N_{\text{live}}) = (5,1)
  \]
- [transcript-backed] Emergency-room case:
  \[
  a_{\text{treat one}} : (N_{\text{live}}, N_{\text{die}}) = (1,5), \qquad
  a_{\text{treat five}} : (N_{\text{live}}, N_{\text{die}}) = (5,1)
  \]
- [transcript-backed] Transplant case:
  \[
  a_{\text{harvest}} : (N_{\text{live}}, N_{\text{die}}) = (5,1), \qquad
  a_{\text{do not harvest}} : (N_{\text{live}}, N_{\text{die}}) = (1,5)
  \]
- [transcript-backed] Dudley-Stevens core defense:
  \[
  \text{better that } 1 \text{ should die so that } 3 \text{ could survive}
  \]
- [standard reconstruction] Dudley-Stevens utilitarian gloss:
  \[
  U(a_{\text{kill Parker}}) \stackrel{?}{>} U(a_{\text{do not kill Parker}})
  \]
- [standard reconstruction] “Greatest good for the greatest number”:
  \[
  \max_a \sum_i u_i(a)
  \]
  Use only as editorial shorthand for Bentham, not as lecture-visible notation.

## Definitions And Objects
- Consequentialist moral reasoning: morality is located in the consequences of an act, in the state of the world that results.
- Categorical moral reasoning: morality is located in absolute requirements, duties, or rights, regardless of consequences.
- Utility: the balance of pleasure over pain; equivalently, happiness over suffering.
- Act \(a\): one available course of action in a case.
- Case variants: trolley driver, bridge/onlooker, emergency doctor, transplant surgeon, Dudley-Stevens lifeboat.
- \(N_{\text{die}}\), \(N_{\text{live}}\): editorial headcount variables for the lecture’s repeated one-versus-many comparisons.
- \(H(a)\): total happiness resulting from act \(a\).
- \(S(a)\): total suffering resulting from act \(a\).
- \(U(a)\): editorial utility shorthand used only in the Bentham section.
- Consent: voluntary agreement of the person whose life is at stake; introduced late in Dudley-Stevens as a possible moral modifier.
- Fair procedure / lottery: agreement to a rule for selecting who will be sacrificed.
- Necessity: the defense that extreme circumstances can excuse or justify what would otherwise be wrong.
- Rights: the possible source of categorical limits against killing, even when utility points the other way.

## Derivation Steps
1. From first judgment to first principle
   1. Set up trolley driver case with two outcomes: kill one or kill five.
   2. Poll the class before naming any theory.
   3. Extract the majority reason: better to kill one than five.
   4. Read this as an initial outcome-based rule.

2. Stress-testing the outcome rule
   1. Keep the arithmetic structure fixed at one versus five.
   2. Change the role from driver to onlooker.
   3. Change the mechanism from steering to pushing.
   4. Observe that the class reaction changes even though the numbers do not.
   5. Conclude that bare outcome-counting does not fully explain the judgments.

3. Medical extension of the same structure
   1. Re-run one-versus-five in emergency-room triage.
   2. Re-run one-versus-five again in organ harvesting.
   3. Note that the same numerical comparison receives different moral responses across cases.
   4. Use that instability to motivate a distinction between consequences and the character of the act.

4. First conceptual extraction
   1. Step back from the cases.
   2. Name the first principle as consequentialist reasoning.
   3. Name the second principle as categorical reasoning.
   4. State the contrast: outcome of the act versus intrinsic quality of the act.

5. Bentham’s utilitarian move
   1. Define utility as pleasure over pain, happiness over suffering.
   2. Treat persons as governed by pain and pleasure.
   3. Infer that morality and legislation should aim at maximizing overall happiness.
   4. Summarize as the greatest good for the greatest number.

6. Dudley-Stevens as utilitarian test case
   1. Present the survival arithmetic: one death may preserve three lives.
   2. Add the wider-effect argument: families and dependents increase the utilitarian stakes.
   3. Let the defense rely on necessity and aggregate welfare.
   4. Register the class’s resistance despite the arithmetic.

7. Splitting the objections
   1. Separate the categorical objection: murder is wrong even if welfare rises.
   2. Separate the procedural objection: perhaps lottery changes the case.
   3. Separate the consent objection: perhaps voluntary agreement changes the case.
   4. End with three open philosophical questions rather than a settled theorem.

## Notation Choices
- Use \(a\) for an act or option throughout.
- Use \(U(a)\) only in the Bentham/utilitarian section.
- Define \(U(a)\) narratively as aggregate happiness minus suffering; do not introduce more elaborate utility machinery.
- Use \(H(a)\) and \(S(a)\) only if the chapter needs a clean shorthand for Bentham’s “happiness over suffering.”
- Use ordered pairs or small tables for case outcomes:
  \[
  (N_{\text{die}}, N_{\text{live}})
  \quad \text{or} \quad
  (N_{\text{live}}, N_{\text{die}})
  \]
  but pick one convention and keep it fixed within a table.
- Recommended convention for consistency: use \((N_{\text{die}}, N_{\text{live}})\) in the trolley-family cases.
- Keep “consequentialist” and “categorical” as primary labels; avoid importing deontic symbols such as \(O(a)\), \(F(a)\), or modal operators.
- Treat “consent,” “lottery,” and “necessity” as labeled conditions or table columns, not as formal operators.
- If a compact slogan is needed, prefer prose plus one simple formula:
  \[
  a^\star = \arg\max_a U(a)
  \]
  and clearly mark it as editorial shorthand.

## Uncertain Mathematics
- No equation or notation is frame-backed; every formula here is transcript-backed or editorial reconstruction.
- Sandel does not write formal symbols, so \(U(a)\), \(H(a)\), \(S(a)\), and \(\arg\max\) are conveniences for the chapter writer, not lecture quotations.
- The lecture never gives a genuine numerical utility calculus; do not add weights, probabilities, discounting, or interpersonal-comparison machinery.
- The line “greatest good for the greatest number” is used as a slogan; it should not be over-read as a precise formal theorem.
- The one-versus-five and one-versus-three structures are stable, but the exact bookkeeping of “who lives” in each scenario is editorially reconstructed from the transcript.
- In Dudley-Stevens, the appeal to families and dependents expands the welfare calculation, but Sandel does not formalize that expansion; keep any extended sum notation cautious.
- The lecture’s real formal payload is comparative case structure plus the tension between maximizing aggregate welfare and respecting categorical limits; do not force it into heavier mathematics than the source can support.