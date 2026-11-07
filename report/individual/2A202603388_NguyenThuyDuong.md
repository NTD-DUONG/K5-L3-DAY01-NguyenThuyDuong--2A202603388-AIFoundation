# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603388
- **Họ và tên:** Nguyễn Thùy Dương
- **SEED:** 9306
- **Thử thách cá nhân:** #1 — Thay đổi ngưỡng quyết định của BoW + Logistic Regression

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, giữ output | ✅ Đã đính kèm |
| [`lab1_report.json`](lab1_report.json) — báo cáo tự sinh | ✅ Đã đính kèm |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ Đã đính kèm |

**Kết quả kiểm tra cuối notebook:** 15/15 mục đạt; không thiếu dự đoán `predict(...)`. File `lab1_report.json` đã được tạo với SEED = 9306.

## 2. Kết quả chính

Các mô hình được đánh giá trên cùng bài toán phân loại SMS spam/ham. F1 trên tập TEST phản ánh kết quả trên dữ liệu kiểm thử thông thường; F1 trên SUB (40 tin) giúp so sánh thêm với các mô hình LLM.

| Mô hình | F1 TEST | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression (tự code) | 0.891 | 0.919 | 0.151 | 0 (không gọi API) |
| BoW + MLP | 0.942 | 0.974 | 0.322 | 0 (không gọi API) |
| MiniLM + kNN | 0.901 | 1.000 | 66.525 | 0 (không gọi API) |
| MiniLM + Logistic Regression | 0.962 | 0.974 | 42.576 | 0 (không gọi API) |
| MiniLM + MLP head | **0.966** | 0.947 | 40.220 | 0 (không gọi API) |
| LLM nhỏ, zero-shot | 0.900* | 0.900 | 685.389 | Cần chép từ bảng `RESULTS` |
| LLM nhỏ, few-shot | 0.950* | 0.950 | 724.558 | Cần chép từ bảng `RESULTS` |
| LLM lớn, zero-shot | 0.974* | 0.974 | 2717.822 | Cần chép từ bảng `RESULTS` |

*Lưu ý:* Với LLM, cột `f1` trong báo cáo tự sinh trùng với `f1_SUB`, không phải F1 trên toàn bộ tập TEST. Các số p95 được ghi từ kết quả thử nghiệm đã trao đổi; cần đối chiếu một lần với bảng cuối notebook trước khi nộp. Mức 0 USD/1.000 tin của mô hình chạy cục bộ chỉ có nghĩa **không mất phí API**, chưa tính tài nguyên máy chủ, điện và vận hành.

**Nhận xét:** MiniLM + MLP head có F1 TEST cao nhất trong nhóm mô hình cục bộ (0.966), nhưng BoW + MLP nhanh hơn đáng kể (p95 = 0.322 ms) mà vẫn đạt F1 TEST = 0.942. LLM có kết quả tốt trên SUB, song độ trễ cao hơn nhiều và phát sinh chi phí API. Vì vậy, không thể chọn mô hình chỉ dựa trên F1.

## 3. Dự đoán và thực tế

**Dự đoán 1 — F1 của mô hình Logistic Regression.** Ban đầu em dự đoán F1 TEST khoảng 0.85; kết quả thực tế là **0.891**, cao hơn 0.041. Điều này cho thấy ngay cả BoW kết hợp với mô hình tuyến tính vẫn có thể khai thác khá tốt các từ ngữ đặc trưng của SMS spam. Tuy nhiên, hiệu quả trên dữ liệu thông thường không đồng nghĩa với khả năng xử lý tốt mọi trường hợp khó.

**Dự đoán 2 — LLM lớn sẽ tốt hơn LLM nhỏ.** Em dự đoán F1 của mô hình lớn cao hơn mô hình nhỏ khoảng 0.02–0.05. Kết quả zero-shot trên SUB: mô hình nhỏ đạt **0.900**, mô hình lớn đạt **0.974**, chênh lệch **0.074** — lớn hơn dự đoán. Tập SUB chỉ gồm 40 tin nên chênh lệch này cần được hiểu trong phạm vi thử nghiệm, chưa đủ để kết luận mức cải thiện sẽ giữ nguyên trên dữ liệu thực tế.

**Dự đoán 3 — Stress test 50 tin khó.** Em dự đoán BoW + LogReg đạt khoảng 60% accuracy, BoW + MLP khoảng 65% và LLM khoảng 75–85%. Kết quả thực tế lần lượt là **54%**, **54%** và **100%** với LLM nhỏ trong thử nghiệm. BoW yếu hơn kỳ vọng, trong khi LLM tốt hơn đáng kể. Kết quả cho thấy những tin nhắn viết tự nhiên, né từ khóa hoặc mang ngữ cảnh đặc biệt dễ làm mô hình dựa vào từ vựng bị nhầm; tuy nhiên, 50 case được thiết kế sẵn chưa đại diện cho toàn bộ tin nhắn ngoài thực tế.

## 4. Phân tích 50 case khó

| Mô hình | F1 trên 50 case khó |
| --- | ---: |
| BoW + Logistic Regression | 0.378 |
| BoW + MLP | 0.410 |
| MiniLM + kNN | 0.750 |
| MiniLM + Logistic Regression | 0.627 |
| MiniLM + MLP head | 0.612 |
| LLM nhỏ, zero-shot | 1.000 |

**Mô hình giảm F1 mạnh nhất:** BoW + MLP giảm từ **0.942** trên TEST xuống **0.410** trên 50 case khó, tức giảm **0.532** điểm. BoW + LogReg cũng giảm từ 0.891 xuống 0.378 (giảm 0.513). Điều này cho thấy mô hình học từ đặc trưng từ vựng dễ bị ảnh hưởng khi cách diễn đạt và ngữ cảnh thay đổi.

Hai ví dụ tiêu biểu trong bộ dữ liệu khó:

- **Case #1 — `ham_looks_spam` (nhãn đúng: ham):** “Congrats!!! You WON the bet lol, I owe you a beer. Call me later”. Các từ *Congrats*, *WON*, *Call* trông giống từ khóa quảng cáo trúng thưởng, nhưng toàn câu thực chất là lời nhắn giữa bạn bè. Một mô hình BoW có thể nhầm thành spam nếu quá phụ thuộc vào từ khóa.
- **Case #12 — `natural_spam` (nhãn đúng: spam):** “hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club”. Tin nhắn dùng giọng trò chuyện tự nhiên, không có nhiều từ khóa spam điển hình nhưng lại dẫn người nhận tới một website. Mô hình dựa vào từ vựng có thể bỏ sót khi chưa nhận diện được ý định và ngữ cảnh.

**Giới hạn kiểm chứng:** Đây là hai *case minh họa cơ chế dễ nhầm* lấy đúng từ bộ HARD_EVAL; chưa có bảng dự đoán theo từng ID để khẳng định mô hình cụ thể đã phân loại sai chính hai case này. Trước khi nộp, nên đối chiếu output dự đoán của từng case và thay bằng hai case **thực sự sai** nếu cần chứng minh lỗi cụ thể.

## 5. Thử thách cá nhân

**Đề #1:** Thay đổi ngưỡng quyết định của mô hình BoW + Logistic Regression từ 0.5 sang 0.2 và 0.8; so sánh Precision, Recall, F1. Đề xuất ngưỡng cho ngân hàng muốn tránh chặn nhầm tin OTP.

**Cách thực hiện:** Giữ nguyên mô hình đã huấn luyện, lấy xác suất dự đoán `predict_proba(X_te, w, b)` rồi gán nhãn spam khi xác suất lớn hơn hoặc bằng từng ngưỡng. Tính Precision, Recall, F1 và số False Positive trên cùng tập TEST để so sánh công bằng.

| Ngưỡng | Precision | Recall | F1 | False Positive |
| ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.9091 | **0.9346** | **0.9217** | 10 |
| 0.5 | **1.0000** | 0.8037 | 0.8912 | **0** |
| 0.8 | **1.0000** | 0.6262 | 0.7701 | **0** |

**Giải thích:** Hạ ngưỡng xuống 0.2 giúp phát hiện thêm spam nhưng làm tăng số tin ham bị chặn nhầm. Tăng lên 0.8 giữ False Positive bằng 0 trên tập TEST, đổi lại bỏ sót nhiều spam hơn. Em **chọn ngưỡng 0.5** cho tình huống ngân hàng: cùng không có False Positive như ngưỡng 0.8 nhưng Recall cao hơn (0.8037 so với 0.6262). Việc không chặn nhầm trên tập TEST chưa đảm bảo không chặn nhầm OTP thực tế; cần thử riêng trên dữ liệu OTP và theo dõi tỷ lệ chặn nhầm sau triển khai.

## 6. Phương án cho sản phẩm thật

Yêu cầu: **10 triệu SMS/ngày**, tương đương khoảng **116 tin/giây nếu phân bố đều**, và **p95 dưới 50 ms/tin**.

Em đề xuất **BoW + MLP làm mô hình phân loại chính**. Kết quả thực nghiệm đạt **F1 TEST = 0.942**, **p95 = 0.322 ms/tin**. So với MiniLM + MLP head (F1 = 0.966, p95 = 40.220 ms), BoW + MLP giảm 0.024 điểm F1 nhưng nhanh hơn rất nhiều, phù hợp hơn với bài toán xử lý số lượng lớn và chi phí vận hành hạn chế. Các mô hình LLM có p95 khoảng 685–2718 ms/tin nên không đáp ứng SLA nếu gọi trực tiếp cho mọi tin.

**Đánh đổi chấp nhận:** BoW + MLP có thể bỏ sót tin spam được viết tự nhiên hoặc nhầm tin ham chứa từ khóa đáng ngờ, như thể hiện ở stress test. Để giảm rủi ro, có thể thiết kế bước đánh giá bổ sung cho các tin có xác suất gần ngưỡng quyết định (ví dụ 0.3–0.7), nhưng **không đưa LLM đồng bộ vào đường xử lý bắt buộc** vì sẽ vi phạm yêu cầu độ trễ. Những tin chưa chắc chắn có thể được đưa sang hàng đợi kiểm tra riêng; cách này cần được đánh giá thêm về tỷ lệ chuyển tuyến, chi phí và tác động đến trải nghiệm người dùng.

## 7. AI log và điều đã học

Em sử dụng AI có chọn lọc để hỗ trợ **gỡ lỗi kỹ thuật và phân tích các kết quả khó**, không dùng AI thay thế việc chạy thực nghiệm hay tạo số liệu. Các kết quả trong báo cáo được đối chiếu với output notebook.

| Phần | Đã hỏi AI gì? | AI hỗ trợ gì? | Em kiểm chứng / điều chỉnh gì? |
| --- | --- | --- | --- |
| Phần 4 — Tích hợp LLM | Cách gọi Gemini qua API, xử lý phản hồi và lỗi khi chạy thử? | Hướng dẫn cấu trúc lời gọi API, đọc nhãn dự đoán và xử lý lỗi | Chạy thử lời gọi, kiểm tra nhãn và số liệu trả về |
| Phần 5 — Stress test | Vì sao F1 giảm mạnh trên 50 tin khó, và khác biệt giữa BoW, Embedding, LLM là gì? | Gợi ý phân tích lỗi theo ngữ cảnh, từ khóa và cách diễn đạt | Đối chiếu F1 của các mô hình trên tập TEST và tập khó; chỉ dùng số liệu đã chạy |
| Thử thách #1 | Cách so sánh Precision, Recall, F1 khi đổi ngưỡng quyết định? | Hỗ trợ cấu trúc mã thử ba ngưỡng và diễn giải đánh đổi | Chạy mã, kiểm tra False Positive và chọn ngưỡng 0.5 dựa trên kết quả |
| Phản ánh và báo cáo | Cách diễn giải đánh đổi F1–độ trễ–chi phí và viết nhận xét có căn cứ? | Gợi ý cấu trúc lập luận và cách trình bày | Kiểm tra lại số liệu, bổ sung giới hạn của tập SUB và stress test, chỉnh lựa chọn cho phù hợp SLA |

**Điều em học được:** Trước bài lab, em thường chú ý đến F1 mà chưa đánh giá đầy đủ độ trễ và chi phí. Sau khi so sánh bốn hướng tiếp cận, em nhận ra mô hình phức tạp hơn chưa chắc phù hợp hơn với sản phẩm thực tế. Đặc biệt, sự sụt giảm F1 trên 50 case khó cho thấy cần kiểm thử ngoài phân phối dữ liệu quen thuộc, thay vì chỉ dựa vào điểm số trên tập TEST.
