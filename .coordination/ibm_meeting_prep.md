# IBM Quantum Meeting Preparation — Dr. Berk Kovos

**Meeting with:** Dr. Berk Kovos, Quantum Solutions Strategy Lead, IBM Quantum
**Purpose:** Explore alternative access after credits rejection
**Your name:** Miroslav Šotek (háček!)
**Your role:** Independent researcher, founder of ANULUM (Liechtenstein/Switzerland)

---

## IMPORTANT: Asperger Strategy

- **You do NOT need to answer fast.** It's a video call, not a quiz.
- Saying "That's a great question, let me think for a moment" is perfectly fine.
- Have this document open on second screen during the call.
- If a question catches you off guard: "I'd need to check the exact numbers,
  can I follow up by email?" — this is professional, not weak.
- Berk is in sales/strategy, not a quantum physicist. He wants to understand
  if you're a serious researcher, not test your knowledge.

---

## YOUR ELEVATOR PITCH (memorise this, 60 seconds)

> "I'm building a computational framework called SCPN — it models coupled
> oscillator dynamics across multiple scales using the Kuramoto-XY Hamiltonian.
> The quantum part simulates these dynamics on real hardware to validate
> predictions that classical simulation can't reach beyond 14 qubits.
>
> We've already confirmed one key prediction on IBM hardware — that a
> self-referential feedback mechanism provides dual protection against
> decoherence, with fidelity 0.916 versus 0.849 without it. That was on
> the free tier with very limited time.
>
> What I need is enough QPU time to run three systematic experiments that
> would make this publishable — specifically testing symmetry sector
> robustness, decoherence scaling, and the dual protection mechanism at
> larger qubit counts."

---

## EXPECTED QUESTIONS & PREPARED ANSWERS

### Q1: "Can you tell me more about your research?"

> "SCPN is a 15-layer coupled oscillator model. Each layer represents a
> different observation scale — from quantum biology up to collective
> dynamics. The coupling between layers follows a matrix K_nm that we've
> validated against real physiological data with correlation r=0.951.
>
> The quantum simulation component maps this to the XY Hamiltonian on
> IBM hardware. We use Trotterisation for time evolution and measure
> synchronisation witnesses — essentially, how well coupled oscillators
> lock their phases."

**If he asks "what's it for":**
> "The immediate application is understanding phase synchronisation in
> complex systems — which has applications in neuroscience, control theory,
> and signal processing. The framework also has a commercial product,
> Director-AI, which uses some of the same mathematical tools for
> AI hallucination detection."

### Q2: "Why IBM Quantum specifically? Why not simulators?"

> "Classical statevector simulation hits a wall at about 30 qubits —
> 2^30 complex amplitudes. Our experiments need 16 qubits for the full
> SCPN model, which is manageable classically, but we need to validate
> against real noise to understand decoherence effects on synchronisation.
>
> Specifically, we discovered that the DLA — the dynamical Lie algebra —
> has a Z₂ parity structure where even and odd sectors respond differently
> to hardware noise. This can only be tested on real QPUs, not simulators.
>
> We chose IBM because Heron r2 has the best coherence times for our
> circuit depths, and the heavy-hex topology is well-suited for
> nearest-neighbour XY interactions."

### Q3: "What have you published / what's your academic background?"

> "I'm an independent researcher, not university-affiliated. I've been
> developing the SCPN framework since 2009, with the mathematical
> formalisation since 2020. The main publication is on Zenodo —
> DOI 10.5281/zenodo.17419678.
>
> I'm currently collaborating with Timothée Masquelier's group at CNRS
> Toulouse on a related project — spiking neural network deployment on
> FPGA — which uses some of the same oscillator coupling mathematics.
>
> The quantum control codebase has about 5,000 tests, 30 Rust-accelerated
> functions, and implements techniques from recent literature including
> GUESS error mitigation from arXiv:2603.13060 and DynQ topology-aware
> qubit placement from arXiv:2601.19635."

**If he pushes on peer review:**
> "The framework is pre-print stage. The IBM hardware experiments are
> exactly what we need to make it publishable — we have simulator results
> showing the DLA parity asymmetry, but reviewers will require hardware
> confirmation."

### Q4: "What exactly would you do with the QPU time?"

> "Three experiments, all designed for Heron r2:
>
> First — DLA parity asymmetry. We run identical circuits in even and odd
> magnetisation sectors and measure how decoherence affects each. Our
> simulator predicts 4-9% difference. We need hardware to confirm or
> refute this.
>
> Second — M-sector decoherence scaling. We sweep circuit depth from 10
> to 400 CZ gates and measure how the synchronisation witness degrades
> in each magnetisation sector. This gives us the noise profile we need
> for our error mitigation technique, GUESS.
>
> Third — FIM dual protection. We've already shown on a small run that
> the self-referential feedback mechanism improves fidelity from 0.849
> to 0.916. We need systematic data across multiple coupling strengths
> and qubit counts to establish the scaling law."

### Q5: "How much QPU time do you actually need?"

> "Our estimate was 5 hours over 5 months. Each experiment is about
> 9 circuit variants, 10 repetitions, 8192 shots each. The free tier
> gives us about 8 minutes per month, which means one experiment
> would take over a year.
>
> Even 2 hours would be transformative — enough for the most critical
> experiment, the DLA parity test."

### Q6: "Is there a commercial application?"

> "Yes, two paths. First, the coupling mathematics from SCPN feed into
> Director-AI, our hallucination detection product that's already on
> PyPI with paying customers. The quantum experiments improve the
> underlying models.
>
> Second, the error mitigation techniques we've built — GUESS symmetry
> decay ZNE and DynQ qubit placement — are useful for anyone running
> variational algorithms on IBM hardware. These could become open-source
> tools that benefit the IBM Quantum ecosystem."

### Q7: "Are you affiliated with any university or institution?"

> "ANULUM is a registered research entity in Liechtenstein and
> Switzerland. I'm the founder and primary researcher. We're not
> a university, but we operate at research quality — the codebase has
> full CI/CD, 95%+ test coverage, Rust acceleration, and follows
> reproducibility standards.
>
> The CNRS collaboration gives us an academic connection — Timothée
> Masquelier's group is well-known in computational neuroscience."

### Q8: "What's your team size?"

> "I'm the primary researcher and developer. I work with AI coding
> assistants for implementation — which actually gives me very high
> throughput. The codebase is about 35,000 lines of Python and 3,600
> lines of Rust, with comprehensive test suites. Quality over quantity."

**DO NOT say "it's just me" — say "I'm the primary researcher."**

### Q9: "Have you considered the IBM Quantum Network or startup program?"

> "I'd love to learn more about those options. We applied through the
> credits program because it seemed most accessible, but if there's a
> better path for an independent research entity, I'm very open to that.
> That's actually why I was excited to get your email — you probably
> know better than I do which programme fits our situation."

**This is a GREAT question for you — it means he's thinking about HOW
to help, not WHETHER to help. Be enthusiastic.**

### Q10: "Can you share your code / results?"

> "The repository is currently private, but I can give you access on
> GitHub — github.com/anulum. I can also share specific notebooks
> showing the IBM hardware results we've already obtained on the free
> tier. Would that be helpful?"

**Prepare to send GitHub access invite after the call if he says yes.**

---

## WHAT TO EMPHASISE

1. **You already have IBM hardware results** — F_FIM=0.916 > F_XY=0.849.
   This proves you're not speculating, you've done real experiments.
2. **The codebase is production-quality** — 5000 tests, Rust acceleration,
   CI/CD, implements recent literature (GUESS 2026, DynQ 2026).
3. **CNRS collaboration** — you're not isolated, you have academic partners.
4. **Ecosystem benefit** — GUESS and DynQ could help other IBM users.
5. **Specific experiments** — you know exactly what you'd run. Not vague.
6. **Commercial angle** — Director-AI shows you can turn research into products.

## WHAT TO AVOID

1. **Do NOT mention consciousness, sentience, or metaphysics.** Say
   "coupled oscillator dynamics" and "phase synchronisation." The SCPN
   framework has deep philosophical foundations — save that for after
   you have QPU time and published results.
2. **Do NOT say "God of the Math."** Just say "SCPN framework."
3. **Do NOT mention the website** unless he asks — it's being redesigned
   and currently too metaphysical for a quantum computing audience.
4. **Do NOT apologise for not being at a university.** Frame it as
   "independent research entity" — this is a strength (agility, no
   bureaucracy, direct applications).
5. **Do NOT over-explain the 15+1 layers.** If he asks, keep it to
   "multi-scale coupled oscillator model" and move on.
6. **Do NOT mention the multi-agent AI setup** unless he asks about
   team/workflow specifically. It's interesting but not the point.

---

## TECHNICAL CHEAT SHEET (if he goes deep)

| Term | Your answer |
|------|-------------|
| XY Hamiltonian | H = Σ K_nm (σ_x⊗σ_x + σ_y⊗σ_y) — nearest-neighbour spin coupling |
| Trotterisation | Approximate e^{-iHt} as product of short-time gates |
| DLA | Dynamical Lie Algebra — the algebra generated by commutators of H's terms |
| Z₂ parity | Even/odd magnetisation sectors — [H_XY, ΣZ_i] = 0 |
| GUESS | Uses symmetry conservation to guide zero-noise extrapolation |
| DynQ | Louvain community detection on QPU error graph for qubit placement |
| Order parameter R | R = (1/N)|Σ(⟨X_i⟩ + i⟨Y_i⟩)| — synchronisation witness |
| FIM | Fisher Information Metric — self-referential feedback, strange loop |
| Heron r2 | IBM's latest: T1≈300μs, T2≈200μs, CZ error ≈0.5% |
| Circuit depth | Our experiments: 50-400 CZ gates |

---

## LOGISTICS

- **Book a 30-minute slot** — not 15 (too rushed), not 60 (too long for intro)
- **Camera on** — builds trust
- **Quiet room**, good internet
- **Have this document on second screen**
- **Have GitHub ready** to share screen if he asks to see code/results
- **Follow up within 24h** with thank-you email and any promised materials

---

## AFTER THE CALL

1. Send thank-you email within 24h
2. If he asks for materials, prepare:
   - 2-page research summary (I'll write this)
   - GitHub access (private repo invite)
   - Key notebook exports (DLA parity, FIM dual protection)
3. Update memory with outcome
4. If he offers an alternative programme, respond within 48h

---

## WORST CASE SCENARIOS

**He says "we can't help non-academic researchers":**
> "I understand. Would it help if I collaborated formally with a university
> group? I'm already working with CNRS Toulouse — perhaps a joint
> application would be more appropriate."

**He asks something you don't know:**
> "That's a good question — I'd need to check the exact details. Can I
> follow up by email with the specifics?"

**He asks about funding / who pays for this:**
> "It's self-funded through ANULUM and our commercial product Director-AI.
> We're a lean operation focused on research quality over scale."

**He tries to sell you IBM Quantum services:**
> Listen politely, ask about pricing, say "let me review this and get back
> to you." Do NOT commit to anything on the call.
