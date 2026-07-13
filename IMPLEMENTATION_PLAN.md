# Implementation Plan — Diode Fitting 

--- 

## Phase A - Single diode model

**Goal:** a validated single-diode IV/JV model.

**Tasks:**
- [ ] Implement single diode equivalent circuit equations
- [ ] Implement light-generated current (linear region)
- [ ] Implement diode curve region + series/shunt resistance effects
- [ ] Implement JV curve generation for both light and dark conditions
- [ ] Implement industry-standard temperature dependence using Faiman/PVsyst as reference
- [ ] Cross-check output against PV Lighthouse's equivalent circuit calculator



---

## Phase B - Fitting functionality

**Goal:** fit the single-diode model to real data such that fitted parameters remain physically meaningful.

**Tasks:**
- [ ] Build data loader for custom/measured IV data, mirroring PV Lighthouse's "add custom data" workflow
- [ ] Implement a baseline fit using an off-the-shelf optimiser (e.g. `scipy.optimize`)
- [ ] Add parameter bounds for all fitted variables
- [ ] Identify which parameter subset matters most physically (discuss with Dong/Mikael if unclear)
- [ ] Design a cost function that heavily penalises unphysical parameter regions 
- [ ] Test for parameter degeneracy: refit from multiple starting points, check whether solutions converge to physically consistent regions
- [ ] Write a test that explicitly checks unphysical parameters are penalised (encodes the core requirement, not just a nice-to-have)

---

## Phase C - Tandem extension

**Goal:** extend everything above to tandem cells (two single-diode circuits in series, 10 parameters).

**Tasks:**
- [ ] Implement tandem circuit model, integrating Mikael's method for tandem description
- [ ] Extend the fitting pipeline to the larger parameter space
- [ ] Extend bounding/penalty logic to the tandem parameter set
- [ ] Validate against a known tandem reference curve if available
- [ ] Investigate interface.materials group's existing work on tandem conductivity impact (see Teams) for cross-reference

---

## Phase D - Circuit solving (SPICE)

**Goal:** move from direct equation-solving to a proper nonlinear circuit solver, benchmarked against TandEx.

**Tasks:**
- [ ] Set up PySpice in Google Colab (workaround for TandEx's macOS porting issue)
- [ ] Reproduce TandEx's circuit setup in PySpice
- [ ] Build a bridge/interface layer so SPICE-based solving can be swapped in for the direct-equation model
- [ ] Benchmark SPICE approach vs. direct-equation approach: speed and accuracy
- [ ] Document why SPICE matters here (fast solving of nonlinear equations introduced by diodes)

---

## Phase E - Paper reproduction

**Goal:** reproduce key results from the reference Joule paper to validate the approach and surface ideas for the unphysicality problem.

**Tasks:**
- [ ] Identify the specific figures/results in the paper to target for reproduction
- [ ] Reproduce using the models/fitting pipeline built in Phases A–C
- [ ] Document where your reproduction matches/diverges from the paper, and why
- [ ] Extract any techniques from the paper for handling parameter degeneracy; feed back into Phase B if improvements are found

---

## Phase F - Deployment

**Goal:** package the modelling/fitting code into a deployable web app.

**Tasks:**
- [ ] Build backend API wrapping model + fitting code 
- [ ] Decide on Python-to-JS wrapper vs. direct JS reimplementation for frontend interactivity
- [ ] Build frontend for IV curve plotting and interactive fitting 
- [ ] Add UI for custom data upload
- [ ] Deploy (target platform TBD — confirm with team whether there's a preferred hosting setup)
- [ ] Write user-facing documentation for the deployed app

---

## Risks / open questions to track

- **Dong's availability** - Phase B's comparison step depends on him; don't block other work waiting for this.
- **Mikael's tandem method** — form/format unclear.
- **TandEx macOS/porting issue** 
- **Physicality metric** — "unphysical" needs a precise, testable definition (e.g. specific parameter ranges backed by literature) before the Phase B penalty can be implemented properly.