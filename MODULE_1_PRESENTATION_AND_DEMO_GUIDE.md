# HƯỚNG DẪN THUYẾT TRÌNH 5 PHÚT & BẢN DEMO MODULE 1 (BIZ 1 & DEV 1)
**Khóa học:** Introduction to Machine Learning (Master Class)  
**Chủ đề chuyên đề:** AI for Medical Treatment (Course 3 - DeepLearning.AI)  
**Phần phụ trách:** Module 1: Treatment Effect Estimation (BIZ 1 & DEV 1 / Hiếu)  
**Tập tin đính kèm đã tạo:**
- 📊 **Slide PowerPoint:** [`Module1_Treatment_Effect_Presentation_5min.pptx`](file:///Users/vophamtuyetnhi/Desktop/BIS/In%20Progress_M9_Introduction%20to%20Machine%20Learning/Group%20Work/Hie%CC%82%CC%81u/Module1_Treatment_Effect_Presentation_5min.pptx)
- 💻 **Ứng dụng Demo Tương tác:** [`treatment_effect_demo.html`](file:///Users/vophamtuyetnhi/Desktop/BIS/In%20Progress_M9_Introduction%20to%20Machine%20Learning/Group%20Work/Hie%CC%82%CC%81u/treatment_effect_demo.html)
- 📓 **Jupyter Notebook:** [`dev1_treatment_effect_estimation.ipynb`](file:///Users/vophamtuyetnhi/Desktop/BIS/In%20Progress_M9_Introduction%20to%20Machine%20Learning/Group%20Work/Hie%CC%82%CC%81u/dev1_treatment_effect_estimation.ipynb)

---

## 1. TỔNG QUAN PHÂN BỔ VÀ CĂN CHỈNH VỚI NHÓM (ALIGNMENT MATRIX)

| Thành viên | Vai trò | Trọng tâm Module 1 & Liên kết | Thời lượng trình bày |
|---|---|---|---|
| **BIZ 1** | Clinical & Statistical Foundation | Thử nghiệm lâm sàng (RCT), Giảm nguy cơ tuyệt đối (ARR), Số bệnh nhân cần điều trị (NNT), Khung Neyman-Rubin $(Y(1), Y(0))$, ATE $\to$ CATE | **1.5 – 2.0 phút** |
| **DEV 1 (Hiếu)** | Machine Learning & Validation | T-Learner vs S-Learner, Hiện tượng co cụm đặc trưng (Shrinkage), Ghép cặp (Matched Pairs), Chỉ số C-for-benefit, Giải thích mô hình SHAP | **2.5 – 3.0 phút** |
| **CẢ HAI** | Live Interactive Demo | Thao tác trực tiếp trên trình duyệt với dữ liệu mô phỏng thời gian thực | **1.5 – 2.0 phút** |

---

## 2. KỊCH BẢN THUYẾT TRÌNH CHI TIẾT (SCRIPT TỪNG GIÂY)

### ⏱️ PHẦN 1: BIZ 1 — NỀN TẢNG Y KHOA & THỐNG KÊ (0:00 – 1:45)

#### Slide 1: Giới thiệu & Định vị đề tài (0:00 – 0:30)
> **Lời nói:** *"Kính thưa Thầy và các bạn, nhóm chúng em xin trình bày Course 3: AI for Medical Treatment. Nếu như Course 1 và Course 2 tập trung vào câu hỏi dự đoán thụ động 'Bệnh nhân có mắc bệnh không?', thì Module 1 của Course 3 giải quyết bài toán can thiệp nhân quả: 'Liệu điều trị này có thực sự giúp bệnh nhân này hay không, và giúp ai nhiều nhất?'. Phần Module 1 sẽ do BIZ 1 trình bày nền tảng thống kê lâm sàng và DEV 1 (Hiếu) trình bày mô hình Machine Learning cá thể hóa cùng bản Demo thực tế."*

#### Slide 2: Thử nghiệm lâm sàng RCT, ARR và NNT (0:30 – 1:15)
> **Lời nói:** *"Trong y khoa, Thử nghiệm ngẫu nhiên có đối chứng (RCT) là tiêu chuẩn vàng vì việc phân nhóm ngẫu nhiên giúp loại bỏ hoàn toàn các yếu tố gây nhiễu (confounders).  
Từ RCT, chúng ta tính được 2 chỉ số cốt lõi:  
1. **ARR (Absolute Risk Reduction):** Mức giảm nguy cơ tuyệt đối. Ví dụ: Nhóm đối chứng có $35\%$ gặp biến cố xấu, nhóm điều trị giảm còn $21\%$, vậy $ARR = 14\%$.  
2. **NNT (Number Needed to Treat):** Số người cần điều trị để cứu 1 người khỏi biến cố: $NNT = 1 / ARR = 1 / 0.14 \approx 7.1$ bệnh nhân.  
Đối với bài toán kinh doanh y tế và bệnh viện, NNT giúp định lượng chính xác chi phí để cứu một mạng người."*

#### Slide 3: Vấn đề cốt lõi của suy luận nhân quả & Bước chuyển sang CATE (1:15 – 1:45)
> **Lời nói:** *"Tuy nhiên, theo mô hình Neyman-Rubin, mỗi bệnh nhân luôn tồn tại 2 kết cục tiềm năng: $Y(1)$ nếu được điều trị và $Y(0)$ nếu không điều trị.  
Vấn đề nan giải nhất của Causal Inference là: **Chúng ta chỉ quan sát được 1 trong 2 kết cục đối với một bệnh nhân có thật; kết cục còn lại (counterfactual) biến mất vĩnh viễn.**  
Hơn nữa, ATE chỉ là trung bình toàn dân số. Trong thực tế, có bệnh nhân hưởng lợi rất nhiều nhưng có người lại bị tác dụng phụ. Do đó, chúng ta cần Machine Learning để ước lượng **CATE (Conditional Average Treatment Effect)** dựa trên đặc trưng $X$ của từng cá nhân. Xin mời DEV 1 trình bày phần kiến trúc mô hình."*

---

### ⏱️ PHẦN 2: DEV 1 (HIẾU) — MÔ HÌNH HÓA CATE, ĐÁNH GIÁ & GIẢI THÍCH (1:45 – 3:30)

#### Slide 4: Kiến trúc Meta-learners: T-Learner vs S-Learner (1:45 – 2:30)
> **Lời nói:** *"Chào Thầy và các bạn, để ước lượng hàm CATE từ dữ liệu RCT, có 2 kiến trúc học siêu cấp (Meta-learners) chính:  
1. **T-Learner (Hai mô hình độc lập):** Chúng ta huấn luyện riêng $\hat{\mu}_1(x)$ trên nhóm điều trị và $\hat{\mu}_0(x)$ trên nhóm đối chứng. Hiệu số $\text{CATE}_T(x) = \hat{\mu}_0(x) - \hat{\mu}_1(x)$. Ưu điểm lớn nhất là mô hình học tự do các mối quan hệ phi tuyến mà không bị giới hạn.  
2. **S-Learner (Một mô hình duy nhất):** Đưa biến điều trị $W$ vào như một đặc trưng bình thường cùng với Tuổi, Huyết áp...  
**Bẫy kỹ thuật nguy hiểm nhất của S-Learner:** Khi dùng Decision Tree hoặc Random Forest, nếu các cây phân nhánh ưu tiên các đặc trưng nguy cơ mạnh (như Huyết áp, Tuổi) mà không bao giờ chọn biến $W$ làm nút chia, thì $\hat{\mu}(x, 1) = \hat{\mu}(x, 0) \implies \text{CATE} = 0$ cho toàn bộ bệnh nhân! Hiện tượng này gọi là **Feature Shrinkage**, khiến mô hình kết luận sai rằng thuốc hoàn toàn vô tác dụng. Vì vậy, nhóm ưu tiên sử dụng T-Learner."*

#### Slide 5: Đánh giá mô hình khi không có nhãn thật: Ghép cặp & C-for-benefit (2:30 – 3:00)
> **Lời nói:** *"Vì counterfactual không quan sát được, làm sao biết mô hình có dự đoán đúng không?  
Giải pháp của Course 3 là **Matched Pairs (Ghép cặp)**: Ghép 1 bệnh nhân nhóm điều trị với 1 bệnh nhân nhóm đối chứng có cùng mức CATE dự đoán để tạo thành một 'cặp bệnh nhân ảo'.  
Từ đó, chúng ta tính chỉ số **C-for-benefit**:  
$$\text{C-for-benefit} = \frac{\text{Số cặp Concordant} + 0.5 \times \text{Risk Ties}}{\text{Tổng số cặp Permissible}}$$  
Trong y văn thế giới, C-for-benefit của bài toán nhân quả thường chỉ đạt **0.55 – 0.65** (khác với AUC chẩn đoán ảnh đạt 0.90+). Mô hình của nhóm đạt **0.6342**, chứng minh khả năng phân loại và ưu tiên bệnh nhân vượt trội so với chọn ngẫu nhiên."*

#### Slide 6: Giải thích mô hình cá thể hóa với SHAP (3:00 – 3:30)
> **Lời nói:** *"Để bác sĩ tin dùng, chúng em tích hợp SHAP từ Module 3 vào mô hình CATE:  
Với một bệnh nhân 62 tuổi có Huyết áp tâm thu 145 mmHg, mức hưởng lợi trung bình quần thể là $+14\%$. Biểu đồ SHAP Waterfall chỉ ra: Huyết áp cao đóng góp $+4.8\%$, Tuổi cao đóng góp $+2.1\%$, BMI đóng góp $+0.9\%$, đưa tổng lợi ích can thiệp lên **$+21.8\%$**. Bác sĩ có thể giải thích trực tiếp lý do kê đơn cho bệnh nhân."*

---

### ⏱️ PHẦN 3: BIZ 1 & DEV 1 — LIVE DEMO TRỰC TIẾP (3:30 – 5:00)

*(Mở file `treatment_effect_demo.html` trên trình duyệt Chrome/Safari)*

1. **BIZ 1 Demo (Tab 1 - 30 giây):**
   - Kéo thanh trượt **Trial Cohort Size** và **Control Event Rate** $\to$ Chỉ vào ô **ARR (+14.0%)** và **NNT (7.1 bệnh nhân)**.
   - Bấm vào bảng **Fundamental Problem of Causal Inference** $\to$ Giải thích hộp $Y(1) = 0$ (quan sát được) vs $Y(0) = ?$ (bị thiếu vĩnh viễn).

2. **DEV 1 Demo (Tab 2, 3, 4 - 1 phút):**
   - **Tab 2 (T-Learner vs S-Learner):** Kéo Huyết áp (SBP) từ 120 lên 160 mmHg $\to$ T-Learner CATE lập tức nhảy lên $+24.5\%$ (High Benefit Badge xuất hiện), trong khi S-Learner chỉ đạt $+14.1\%$ do hiện tượng shrinkage.
   - **Tab 3 (Matched Pairs & C-for-benefit):** Cho thầy xem bảng ghép cặp với $y_d \in \{+1, 0, -1\}$ và điểm số **0.6342** trên dải Benchmark.
   - **Tab 4 (SHAP Waterfall):** Cho thầy xem thanh phân rã từng đặc trưng SBP, Age, BMI và kết luận đơn thuốc cá thể hóa.

---

## 3. BỘ CÂU HỎI & TRẢ LỜI PHẢN BIỆN DÀNH CHO GIẢNG VIÊN (Q&A CHEAT SHEET)

#### ❓ Câu hỏi 1: "Vì sao không dùng thẳng một mô hình dự đoán nguy cơ thông thường mà phải làm Causal Inference?"
> **Trả lời:**  
> Mô hình dự đoán nguy cơ chỉ học mối tương quan $P(Y \mid X)$. Trong thực tế lâm sàng, những bệnh nhân nặng thường được bác sĩ ưu tiên cho dùng thuốc nhiều hơn. Nếu dùng ML thông thường, mô hình sẽ học ngược lại rằng *"Ai uống thuốc thì nguy cơ tử vong càng cao"*. Causal Inference thông qua RCT và CATE giúp cô lập chính xác tác động can thiệp độc lập của thuốc, trả lời câu hỏi *"Nếu cho thuốc thì nguy cơ giảm bao nhiêu?"*.

#### ❓ Câu hỏi 2: "Chỉ số C-for-benefit chỉ đạt 0.63 thì có dùng được trên thực tế không?"
> **Trả lời:**  
> **Hoàn toàn dùng rất tốt**. Trong bài toán nhân quả, nhãn thật (counterfactual) không bao giờ quan sát được trực tiếp, nên độ nhiễu nội tại rất lớn. Các nghiên cứu lâm sàng công bố quốc tế về CATE thường chỉ đạt mức 0.55 – 0.65. Mức 0.6342 có nghĩa là khi so sánh 2 cặp bệnh nhân, mô hình xếp hạng đúng người hưởng lợi nhiều hơn trong $63.4\%$ trường hợp, mang lại giá trị phân bổ nguồn lực y tế vượt trội so với quyết định kinh nghiệm thông thường ($50\%$).

#### ❓ Câu hỏi 3: "Khi nào nên dùng T-Learner, khi nào nên dùng S-Learner?"
> **Trả lời:**  
> - **S-Learner:** Phù hợp khi cỡ mẫu rất nhỏ (dưới vài trăm mẫu) vì nó tận dụng toàn bộ dữ liệu để học hàm nguy cơ nền tảng. Tuy nhiên, nhược điểm chí mạng là biến điều trị $W$ dễ bị co cụm hệ số về 0 (shrinkage).  
> - **T-Learner:** Phù hợp khi cỡ mẫu thử nghiệm đủ lớn ($N \ge 1,000$). Nó loại bỏ hoàn toàn rủi ro shrinkage và nắm bắt được các tương tác phi tuyến phức tạp giữa thuốc và cơ địa từng bệnh nhân.

#### ❓ Câu hỏi 4: "Mối liên hệ giữa Module 1 và Module 3 (SHAP) là gì?"
> **Trả lời:**  
> Module 1 cho chúng ta con số hiệu quả điều trị $\text{CATE}(x)$, nhưng đây vẫn là 'hộp đen'. Bác sĩ tại giường bệnh sẽ không bao giờ đồng ý kê đơn thuốc đặc trị nếu không biết *tại sao* bệnh nhân này lại được khuyên dùng. Giá trị Shapley từ Module 3 phân rã chính xác đóng góp biên của từng chỉ số xét nghiệm, biến kết quả ML thành giải thích y khoa thuyết phục.
