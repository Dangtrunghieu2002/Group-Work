# 🎙️ Master Presentation Scripts: AI for Medical Treatment Demo

> **Total Duration:** ~20 minutes (~4–5 minutes per speaker + 2 minutes Q&A)  
> **Course:** M9 - Introduction to Machine Learning (Group Project)  
> **Topic:** AI for Medical Treatment: From Clinical Foundations to Precision Medicine & Deep Vision  
> **Speakers:**  
> 1. **Trần Thị Bảo Khương (ID: 20225027)** — Part 1: Clinical Trials, RCTs & The Causal Dilemma  
> 2. **Đặng Trung Hiếu (ID: 20225003)** — Part 2: Individualized CATE, T-Learner vs S-Learner & SHAP Receipts  
> 3. **Lê Ngọc Thuỳ Dung (ID: 20225030)** — Part 3: 100k PACS Label Mining & 14-Disease DenseNet121  
> 4. **Lê Khánh Linh (ID: 20225020)** — Part 4: Medical NLP (BioBERT) & Vision Explainability (Grad-CAM)  

---

## 🌟 Opening & Overview (Khương or Hiếu - 1 Minute)

**[👉 Screen Action: Start on "Overview" Tab]**

> *"Good morning Professor and fellow classmates. Welcome to our group presentation on **AI for Medical Treatment**.*  
>  
> *Modern healthcare faces a major dilemma: clinical trials tell us whether a drug works on **average**, but doctors treat **individual patients**, not averages. Furthermore, hospitals possess millions of unstructured X-ray reports and scans, but transforming this big data into reliable, explainable clinical decisions requires a complete end-to-end AI pipeline.*  
>  
> *Today, our team—Khương, Hiếu, Dung, and Linh—will walk you through our unified 4-stage architecture, covering causal RCT foundations, individualized machine learning (CATE), hospital PACS mining, and deep vision explainability.*  
>  
> *Let us begin with Part 1: Clinical Trials and Population Foundations."*

---

## 👩‍💼 Part 1: Clinical Foundations, RCT & The Causal Dilemma
**Speaker:** Trần Thị Bảo Khương (ID: 20225027)  
**Time Allocation:** 4.5 Minutes  
**Focus:** Randomized Controlled Trials (RCT), Absolute Risk Reduction (ARR), NNT, and Neyman-Rubin Potential Outcomes.

---

### Chapter 1: Clean RCT vs Confounded Doctor Choice (2.5 mins)

**[👉 Screen Action: Click tab "Khương" &bull; Flow Step 1 "Trial RCT & ARR" is active]**

> *"Thank you. In medical science, the gold standard for testing a new drug is the **Randomized Controlled Trial (RCT)**. But why can't we simply look at observational hospital records?"*

**[👉 Screen Action: Click `👨‍⚕️ Doctor Choice (Biased)` button]**

> *"Look at what happens when doctors choose who gets the drug based on intuition. Sicker, elderly patients are given the drug more often. This opens a **backdoor confounding path** from Patient Age and SBP to the Treatment allocation ($X \rightarrow W$).*  
>  
> *As you can see in **Card 5 (Expected Output)**, the Standardized Mean Difference (SMD) for Blood Pressure explodes to $+0.45$. The observational data is confounded: doctors cannot distinguish whether poor outcomes were caused by the disease or the drug itself!"*

**[👉 Screen Action: Click `🎲 Coin Flip (Clean RCT)` button & drag "Trial Patients N" slider to 1,000]**

> *"Now, when we enforce a randomized coin flip, we physically sever the backdoor arrow ($X \not\rightarrow W$). The SMD drops to $0.02$—perfect covariate balance.*  
>  
> *Under clean RCT conditions, our live math in **Card 4** shows:*  
> $$\text{ARR} = P(Y=1 \mid W=0) - P(Y=1 \mid W=1) = 20.0\% - 6.0\% = +14.0\%$$  
> $$\text{NNT} = \frac{1}{\text{ARR}} = \frac{1}{0.14} = 7.1 \text{ patients}$$  
>  
> *In **Card 6 (Visualizer)**, you can see our dynamic cohort: for every $7.1$ patients treated with this drug, exactly **1 human life is saved** from a major stroke event ($🟢$)."*

---

### Chapter 2: The Parallel Universe Dilemma & Missing Counterfactuals (2 mins)

**[👉 Screen Action: Click Flow Step 2 "Missing Worlds & CATE"]**

> *"While an ARR of $+14.0\%$ proves the drug works for the **population**, it creates a profound dilemma for the individual doctor sitting with a real patient.*  
>  
> *According to the **Neyman-Rubin Causal Model**, each individual has two potential outcomes: $Y_i(1)$ (outcome if treated) and $Y_i(0)$ (outcome if untreated)."*

**[👉 Screen Action: Click `Grandpa An (Patient #42)` button in Card 2]**

> *"For Grandpa An (Patient #42), we observed him taking the drug ($W=1$) and surviving ($Y=0$). But what would have happened to him in the parallel universe where he received only a placebo ($W=0$)?*  
>  
> *As shown in **Card 4 (Math)**, his counterfactual $Y(0)$ is fundamentally **unobservable and missing** in the real world.*  
>  
> *Treating every patient with a one-size-fits-all $+14\%$ assumption is dangerous: young low-risk patients may receive unnecessary side effects, while high-risk elderly patients need aggressive care.*  
>  
> *To solve this missing counterfactual puzzle, we need machine learning to predict the **Conditional Average Treatment Effect (CATE)** for individual clinical twins. I will now hand over to **Hiếu** to demonstrate how our CATE models achieve this precision."*

---

## 👨‍💻 Part 2: Individualized CATE, T-Learner vs S-Learner & SHAP Receipts
**Speaker:** Đặng Trung Hiếu (ID: 20225003)  
**Time Allocation:** 4.5 Minutes  
**Focus:** Baseline Risk Profiling, T-Learner vs S-Learner Architecture, Matched Twin Validation ($C=0.6342$), and TreeSHAP.

---

### Chapter 1: Personalized Risk Profiling & CATE (1.5 mins)

**[👉 Screen Action: Click tab "Hiếu" &bull; Flow Step 1 "Individual CATE" is active]**

> *"Thank you Khương. Picking up from the parallel universe dilemma, machine learning allows us to estimate each patient's individual benefit by conditioning on their biological features $X = \{\text{Age, SBP, BMI}\}$:*  
> $$\text{CATE}(X) = \mathbb{E}[Y(0) - Y(1) \mid X]$$  
>  
> *Let us test two contrasting patient profiles."*

**[👉 Screen Action: Click preset `Alex (Young)` &bull; then click preset `Grandpa An` in Card 2]**

> *"Notice the dramatic difference:*  
> *• For **Alex** (Age 35, SBP 118), his baseline untreated risk is only $5.4\%$. The drug reduces his risk to $3.4\%$—a marginal benefit of just $+2.0\%$. Prescribing expensive medication with potential side effects here is bad medicine; lifestyle changes come first.*  
>  
> *• But for **Grandpa An** (Age 70, SBP 165, BMI 31), his untreated stroke risk is **$48.2\%$**. With treatment, his risk plummets to **$21.8\%$**—yielding a massive **$+26.4\%$ Net Benefit**!*  
>  
> *Grandpa An benefits **13 times more** than Alex. That is precision AI medicine in action."*

---

### Chapter 2: The Model Battle: T-Learner vs S-Learner (1.5 mins)

**[👉 Screen Action: Click Flow Step 2 "T vs S Learner" &bull; Click `🌲 S-Learner (Shrinkage)` button]**

> *"How should we structure the underlying machine learning model?*  
>  
> *The naive approach is an **S-Learner** (Single Learner), where treatment $W$ is simply fed as an extra feature alongside Age and SBP into one large decision tree.*  
>  
> *As you can see in **Card 6 (Visualizer)**, the tree prioritizes high-variance features like SBP and Age at top splits, severely penalizing the treatment feature. Grandpa An's estimated benefit artificially collapses from $+26.4\%$ down to **$+15.2\%$** due to **Regularization Shrinkage Bias**."*

**[👉 Screen Action: Click `🌲🌲 T-Learner (2 Doctors)` button]**

> *"To fix this, we deploy a **T-Learner** (Two Learners). We train **two independent models**: Doctor 1 $\mu_1(X)$ trained exclusively on treated patients, and Doctor 2 $\mu_0(X)$ trained on control patients.*  
>  
> *By subtracting their predictions ($\mu_0(X) - \mu_1(X)$), T-Learner completely eliminates shrinkage bias and recovers the full, unbiased **$+26.4\%$ benefit** for Grandpa An!"*

---

### Chapter 3: Clinical Twin Validation & SHAP Itemized Receipts (1.5 mins)

**[👉 Screen Action: Click Flow Step 3 "Twin Test & SHAP" &bull; Click `🎲 Resample Clinical Twin Pairs` button]**

> *"Since real patients cannot have cloned counterfactuals, how do we evaluate our CATE model in hospital practice?*  
>  
> *We construct **statistical twin pairs**—matching treated and control patients with identical Age and SBP. As shown in **Card 5**, we evaluate the model using the **C-for-benefit metric**:*  
> $$C = \frac{\text{Concordant} + 0.5 \times \text{Ties}}{\text{Permissible}} = 0.6342$$  
> *$0.60\text{--}0.65$ is the internationally recognized gold standard for causal treatment models (equivalent to ~0.80 AUC in standard classification).*  
>  
> *Finally, in **Card 6**, we provide doctors with an **Itemized SHAP Push Receipt**:  
> • Baseline Drug: $+4.0\%$  
> • SBP 165 push: $+15.2\%$  
> • Age 70 push: $+7.8\%$  
> • Total CATE: **$+26.4\%$**.*  
>  
> *Doctors understand exactly WHY the AI recommends treatment.*  
>  
> *Now, where do we get the large-scale hospital ground truth data to power these models? I pass the presentation to **Dung** to reveal our 100k PACS radiology mining pipeline."*

---

## 👩‍💼 Part 3: 100k PACS Big Data Mining & 14-Disease Vision CNN
**Speaker:** Lê Ngọc Thuỳ Dung (ID: 20225030)  
**Time Allocation:** 4.5 Minutes  
**Focus:** PACS Radiology Mining, NegBio Rule Parsing, 14-Disease DenseNet121 CNN & F1 Harmonic Mean Optimization.

---

### Chapter 1: Mining 100,000 Unstructured PACS Reports (1.5 mins)

**[👉 Screen Action: Click tab "Dung" &bull; Flow Step 1 "100k PACS Labeling" is active]**

> *"Thank you Hiếu. To train deep learning models in healthcare, we cannot afford to pay radiologists millions of dollars to manually label hundreds of thousands of X-rays.*  
>  
> *Instead, our system connects directly to hospital **PACS (Picture Archiving and Communication Systems)** to mine free-text radiology notes into a structured binary label matrix ($Y \in \{0, 1\}^{100,000 \times 14}$). Let us examine how NegBio parses complex medical syntax."*

**[👉 Screen Action: Click `Case #01 (Negation)` &bull; then `Case #03 (Synonym Mapping)` in Card 2]**

> *"In **Case #01**, the radiologist wrote: *'No evidence of active pneumonia'*. Standard keyword search would falsely flag Pneumonia. NegBio uses dependency graph parsing to identify the negation scope, correctly outputting **Label = 0 (Negated)**.*  
>  
> *In **Case #03**, the report states *'airspace consolidation consistent with bronchopneumonia'*. NegBio maps clinical ontology synonyms via UMLS, correctly setting **Pneumonia = 1 (Positive)**.*  
>  
> *This automated pipeline generates 100,000 ground truth labels at **$0 cost** and **100% HIPAA compliance** behind hospital firewalls."*

---

### Chapter 2: 14-Disease DenseNet121 Vision CNN & F1 Tuning (2 mins)

**[👉 Screen Action: Click Flow Step 2 "14-CNN & F1 Tuning"]**

> *"Next, we feed the Chest X-Rays and mined labels into a **14-Disease DenseNet121 CNN architecture**."*

**[👉 Screen Action: Click `Scan #0599 (Pneumonia)` image &bull; then click `Scan #1042 (Cardiomegaly)` image]**

> *"Notice our multi-label sigmoid design:*  
> *• When we feed **Scan #0599**, the network identifies right lower lobe consolidation, outputting **$88\%$ Pneumonia** and **$72\%$ Infiltration** simultaneously.*  
> *• When we feed **Scan #1042**, the network isolates an enlarged cardiac silhouette ($>55\%$ CTR), outputting **$94\%$ Cardiomegaly** with clean lungs.*  
>  
> *Because rare chest pathologies represent less than $1\%$ of hospital data, a naive model predicting 0 for all cases achieves $99\%$ accuracy while killing sick patients!"*

**[👉 Screen Action: Drag "Operating Decision Threshold" slider from 0.50 to 0.70 & back to 0.50]**

> *"To solve class imbalance, our math engine in **Card 4** evaluates the **F1-Score Harmonic Mean**:*  
> $$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 0.923$$  
> *By tuning the decision threshold to $0.50$, we achieve an optimal balance, ensuring zero dangerous false negatives for high-mortality conditions."*

---

### Chapter 3: 2026 Frontier: Vision-Language Models (VLM) (1 min)

**[👉 Screen Action: Click Flow Step 3 "2026 Frontier Vision" &bull; Click `2026: Vision-Language Model` button]**

> *"Looking forward to 2026, the frontier is shifting from two-stage rule parsing to **Multimodal Vision-Language Models like BiomedCLIP**.*  
>  
> *Instead of hand-engineered regex patterns, contrastive embeddings align radiological text directly with image patches, enabling zero-shot disease classification.*  
>  
> *However, can clinicians trust these deep networks without opening the black box? I now hand over to **Linh** to demonstrate our Natural Language & Vision Explainability labs."*

---

## 👨‍💻 Part 4: Medical NLP (BioBERT) & Explainability Labs (Grad-CAM)
**Speaker:** Lê Khánh Linh (ID: 20225020)  
**Time Allocation:** 4.5 Minutes  
**Focus:** BioBERT WordPiece Tokenizer & Attention Mask, Layer 424 Grad-CAM Localization, and Permutation Feature Importance.

---

### Chapter 1: Medical NLP Preprocessing & BioBERT Attention (2 mins)

**[👉 Screen Action: Click tab "Linh" &bull; Flow Step 1 "BioBERT Attention" is active]**

> *"Thank you Dung. Clinical radiology text is notoriously messy—filled with shorthand, typos, and formatting noise. Before feeding text into modern transformers like BioBERT, robust text normalization is mandatory."*

**[👉 Screen Action: Change dropdown to preset 2 `"Forxiga (dapagliflozin)..."` &bull; Click `✨ Clean Medical Text`]**

> *"Our cleaner standardizes punctuation, handles medical conjunctions, and maps terms into **WordPiece Token IDs**.*  
>  
> *As displayed in **Card 5 & 6**, BioBERT formats the input into a fixed tensor shape $(1, 60)$:  
> • $[CLS]$ classification token: ID $101$  
> • Sub-word tokens like `dapagliflozin` ($14210$)  
> • $[SEP]$ boundary token: ID $102$  
> • Zero-padding tokens $[PAD]$: ID $0$.*  
>  
> *Our Attention Mask $M_i$ sets real tokens to $1$ and padding tokens to $0$, allowing the multi-head self-attention mechanism to focus exclusively on clinical semantics."*

---

### Chapter 2: Grad-CAM Localization & Permutation Importance (2.5 mins)

**[👉 Screen Action: Click Flow Step 2 "Grad-CAM & Permutation"]**

> *"Finally, we address the critical question asked by every doctor: **'Can we trust what the AI sees?'**  
>  
> *We implement **Grad-CAM (Gradient-Weighted Class Activation Mapping)** on the final convolutional layer (Layer 424) of our DenseNet121 model."*

**[👉 Screen Action: Click `🫁 Pneumonia` &bull; Adjust Opacity Slider to 85% &bull; Then click `❤️ Cardiomegaly`]**

> *"Watch the heatmap in **Card 6 (Visualizer)**:*  
> *• When evaluating for **Pneumonia**, the backpropagated gradients $\alpha_k^{\text{pneu}}$ isolate the exact inflammatory opacity in the **right lower lung field** ($🟢 \rightarrow 🔴$).*  
> *• When evaluating for **Cardiomegaly**, the gradients dynamically shift to outline the **expanded cardiac silhouette borders**.*  
>  
> *This ensures the AI is diagnosing real pathology rather than cheating on hospital metal markers or image artifacts."*

**[👉 Screen Action: Click `🔀 Shuffle Age` in Card 2]**

> *"To explain tabular risk factors, we perform **Permutation Feature Importance**.*  
>  
> *As shown in **Card 5**, when we randomly shuffle Patient Age across the test set, model discrimination drops by **$-0.2228$ C-Index**—proving that Age is the \#1 biological driver of stroke risk, followed by Blood Pressure.*  
>  
> *Through BioBERT, Grad-CAM, and Permutation Importance, our clinical pipeline is **100% transparent, accountable, and clinician-ready**.*  
>  
> *Let us now conclude with our Wrap-Up and Q&A session."*

---

## 👥 Part 5: Strategic Wrap-Up & Group Q&A (All Members - 2 Minutes)

**[👉 Screen Action: Click tab "Wrap-Up"]**

### Summary & Core Takeaways (Hiếu / Khương):
> *"To summarize our work:*  
> 1. **Khương** established causal foundations, showing how RCT coin flips cut confounding bias and why ATE population averages fail individual patients.  
> 2. **Hiếu** built precision CATE models using T-Learners to protect against tree shrinkage, validating benefit on twin cohorts ($C=0.6342$) with SHAP receipts.  
> 3. **Dung** auto-labeled 100,000 PACS reports with NegBio and trained a 14-disease DenseNet121 vision CNN optimized for F1 harmonic balance.  
> 4. **Linh** opened the black box, proving token attention structures and visual Grad-CAM pathology localization.  
>  
> *Together, this bridges the gap between raw medical big data and individualized, lifesaving clinical decisions.*  
>  
> *We are now ready to take questions from the Professor and the audience. Thank you!"*

---

### 💡 Frequently Asked Professor Questions (Quick Answer Guide)

| Question | Speaker | Key 30-Second Answer |
| :--- | :--- | :--- |
| **Q1. Why not use standard risk models instead of causal AI?** | **Khương / Hiếu** | *"Risk models only predict who is sick (correlation); causal AI tells us who will actually improve with the drug (causality), preventing unnecessary medication in non-responders."* |
| **Q2. Is C-for-benefit of 0.6342 high enough for clinical deployment?** | **Hiếu** | *"Yes! In causal ML, 0.60–0.65 is the international hospital gold standard because unobservable counterfactuals add natural twin matching variance (equivalent to ~0.80 standard AUC)."* |
| **Q3. Why is T-Learner preferred over S-Learner in medicine?** | **Hiếu** | *"S-Learner suffers from shrinkage bias—decision trees split on Age/BP and ignore treatment $W$. T-Learner trains 2 separate models to capture 100% of treatment interactions."* |
| **Q4. Does Grad-CAM guarantee medical correctness?** | **Linh** | *"No. Grad-CAM shows WHERE the network looked, acting as a vital safety filter so clinicians can verify the AI isn't hallucinating on image artifacts."* |
| **Q5. Why use rule-based NLP if Large Language Models (LLMs) exist?** | **Dung / Linh** | *"Hospital PACS contain millions of records behind firewalls. Rule-based NegBio is 100% HIPAA-compliant, zero-cost, runs locally at lightning speed, and requires zero GPUs."* |
| **Q6. If rule labels contain noise, how does the CNN reach 92% F1?** | **Dung** | *"Big data scale filters out noise. Across 100,000 X-rays, random label errors cancel out, allowing the deep CNN to learn true physical visual disease features."* |

---
