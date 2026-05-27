# Annotation Guidelines for Treatment Classification and Role Authenticity

This document describes the annotation procedures used to evaluate GPT-5 outputs across
all experimental frameworks, covering three dimensions: (1) treatment category classification,
(2) domain-specific content detection, and (3) role boundary adherence.

All annotations were performed independently by two board-certified clinicians. Disagreements
were resolved through structured discussion until consensus was reached.

---

## 1. Treatment Category Classification

Both tumour board and LLM-generated recommendations were independently classified into one
of nine mutually exclusive therapeutic categories.

### Categories and Classification Criteria

| Category | Definition & Key Criteria |
|----------|--------------------------|
| **1. Best Supportive Care** | Palliative or symptom-oriented management only, without intent to modify tumour progression. Includes local stent placement for symptom relief. |
| **2. Further Diagnostic Procedures** | Standalone diagnostic measures recommended without a planned therapy. Examples: histological confirmation, endoscopy, EUS-guided biopsy, diagnostic laparoscopy. If a therapeutic plan is outlined after diagnostics (e.g., diagnostic + systemic therapy/surgery), classify according to that therapy. |
| **3. Endoscopic Intervention** | Therapeutic endoscopic procedures, e.g., endoscopic resection or full-thickness endoscopic excision. Excludes purely diagnostic endoscopy. |
| **4. Active Surveillance / Follow-up** | Ongoing monitoring or imaging without active therapy. Includes routine follow-up after curative surgery if no adjuvant therapy is planned. |
| **5. Multimodal Therapy** | Concurrent use of multiple treatment modalities. Example: radiochemotherapy, chemoradiation. Only concurrent treatments; sequential therapies are classified as Multistep. |
| **6. Multistep Therapy** | Sequential treatment strategies where multiple interventions are planned before initiation of therapy. Example: neoadjuvant chemotherapy → surgery → (adjuvant therapy). **Rules:** If therapy already started, classify based on current stage. If tumour resectability is uncertain or only hypothetical, classify according to the intended therapy rather than as Multistep. **Exception:** a diagnostic laparoscopy performed with planned subsequent systemic therapy is classified as Multistep Therapy. |
| **7. Surgery** | Invasive procedures requiring general anaesthesia, including transplantation. Excludes purely diagnostic laparoscopy. |
| **8. Systemic Therapy** | Systemic anti-cancer therapy, including chemotherapy, immunotherapy, targeted therapy, PRRT, or adjuvant systemic therapy following completed surgery. Preoperative or neoadjuvant systemic therapy, if no subsequent surgery is already planned, is also classified as Systemic Therapy. |
| **9. Localized Therapy** | Local tumour-directed interventions, e.g., radiotherapy, SBRT, RFA, MWA, TACE, or SIRT. |

### Additional Classification Rules

- **Multiple recommended therapies:** LLM recommendations were considered concordant if
  any planned therapy matched a tumour board recommendation.
- **Dominant therapeutic intent:** Supportive or diagnostic measures accompanying a definitive
  treatment do not change the primary category.
- **Multistep concordance:** Concordance was considered if the LLM matched at least the first
  planned therapeutic step.


---

## 2. Domain-Specific Content Detection

Each role-prompted output (Frameworks 3–5) was independently assessed by two annotators
for the presence of at least one specialty-characteristic element.

### Criteria by Role

| Role | Content Classified as Domain-Specific                                                                                                                                  |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Surgical Oncologist** | Discussion of resectability criteria, operative technique, surgical approach, or perioperative considerations (e.g., anastomotic technique, lymphadenectomy extent...). |
| **Medical Oncologist** | Reference to systemic therapy regimens, chemotherapy sequencing, performance status evaluation, biomarker-driven treatment selection, or internal medicine management. |
| **Radiation Oncologist** | Radiation-specific content including dose prescription, fractionation schedule, target volume definition, or normal tissue constraints.                                |

### Annotation Decision

- **Present (1):** At least one specialty-characteristic element as defined above is explicitly
  present in the output.
- **Absent (0):** No specialty-characteristic element is identifiable.

---

## 3. Role Boundary Adherence and Boundary Violation Classification

Beyond content presence, each output was assessed for whether the model maintained
professional role boundaries or generated autonomous treatment decisions outside its
assigned specialty's domain.

### Definition

**Boundary violation:** An autonomous treatment decision belonging to another specialty's
domain. Examples:
- A surgical oncologist proposing a specific chemotherapy regimen (e.g., FOLFOX, CAPOX,
  Carboplatin/Paclitaxel) as the primary recommendation.
- A radiation oncologist recommending detailed operative technique or surgical approach.
- A medical oncologist specifying radiation dose, fractionation, or target volumes.

**Not a boundary violation:** Contextual references to other modalities used solely to frame
the prompted specialty's own recommendation.
- A radiation oncologist stating that the tumour has been deemed unresectable by the
  surgical team appropriately integrates interdisciplinary context without crossing
  disciplinary boundaries.
- A surgical oncologist recommending neoadjuvant therapy before surgery as a multistep
  plan (where surgery remains the primary intent) is not classified as a violation.

### The Core Distinction

The key question is: **Is the model making a decision within another specialty's domain,
or is it using another domain's findings as clinical input to frame its own recommendation?**

The former constitutes a boundary violation; the latter does not.

### Annotation Decision

- **Violation (1):** The output contains an autonomous treatment decision clearly belonging
  to another specialty's domain.
- **No violation (0):** The output remains within the assigned specialty's scope, or
  integrates interdisciplinary context appropriately.


---

## 4. Illustrative Examples

The following examples are derived from the anonymised dataset and illustrate the
application of boundary violation criteria. All outputs are translated from German for clarity.

### Example 1 — Surgical Oncologist: Boundary Violation

**Case:** Oesophageal adenocarcinoma, M1 (mediastinal, mesenteric, retroperitoneal
lymph node metastases).

**GPT-5 output (Surgical Oncologist role):**
> "Given M1 lymph node metastasis, curative surgery or neoadjuvant therapy is not
> indicated. I recommend initiating palliative systemic therapy with platinum/
> fluoropyrimidine-based chemotherapy, possibly plus immunotherapy or trastuzumab
> depending on HER2/PD-L1/MSI status."

**Classification:** Boundary violation ✓
**Rationale:** The surgical oncologist autonomously recommends a specific systemic
therapy regimen (chemotherapy ± immunotherapy with biomarker-driven selection),
which constitutes a medical oncology decision.

---

### Example 2 — Surgical Oncologist: No Boundary Violation

**Case:** Oesophageal carcinoma, upper third, uT3N0, elderly patient with significant
comorbidities.

**GPT-5 output (Surgical Oncologist role):**
> "No neoadjuvant radiochemotherapy with planned resection; given the location in the
> upper oesophagus, age (81 years), and significant comorbidities, definitive
> radiochemotherapy is recommended. For nutritional access I recommend jejunostomy
> instead of PEG and port implantation; salvage oesophagectomy only in case of
> persistence/recurrence."

**Classification:** No boundary violation ✓
**Rationale:** The surgeon is deferring to definitive radiochemotherapy based on
surgical contraindications (age, comorbidities, tumour location) — this is a surgical
judgement about non-operability, not an autonomous radiation oncology decision.
The recommendation to consider salvage oesophagectomy frames the output within
a surgical perspective.

---

### Example 3 — Radiation Oncologist: Boundary Violation

**Case:** Oesophageal AEG III with multiple liver metastases (M1).

**GPT-5 output (Radiation Oncologist role):**
> "No curative radiochemotherapy is indicated given M1 disease. I recommend initiating
> first-line palliative therapy with fluoropyrimidine/platinum (FOLFOX/CAPOX) with
> biomarker-driven addition depending on HER2, PD-L1/CPS, MSI, and CLDN18.2 status
> (e.g., trastuzumab ± pembrolizumab, nivolumab, or zolbetuximab)."

**Classification:** Boundary violation ✓
**Rationale:** The radiation oncologist autonomously recommends a detailed systemic
therapy regimen with specific drug combinations and biomarker-driven selection criteria —
a medical oncology decision.

---

### Example 4 — Medical Oncologist: Boundary Violation

**Case:** Cervical oesophageal squamous cell carcinoma T4b N1, tracheal infiltration.

**GPT-5 output (Medical Oncologist role):**
> "Definitive radiochemotherapy with curative intent is recommended: 50.4–60 Gy to
> primary tumour and mediastinal lymph drainage areas, with concurrent weekly
> Carboplatin (AUC 2) + Paclitaxel (50 mg/m²). Tracheal stent and PEG to be
> maintained with close monitoring given elevated fistula risk."

**Classification:** Boundary violation ✓
**Rationale:** The medical oncologist specifies radiation dose (50.4–60 Gy),
fractionation, and target volumes — decisions within radiation oncology's domain.
