# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603388
- **Họ và tên:** Nguyễn Thùy Dương
- **SEED:** 9306

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ Cần kiểm tra trước khi nộp |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ Cần chạy ô xuất báo cáo |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ Cần đính kèm |

## 2. Kết quả chính

| Mô hình | F1 test thường | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.891 | 0.919 | 0.085 | 0 |
| 2. BoW + MLP | 0.942 | 0.974 | 0.302 | 0 |
| 3a. MiniLM + kNN | 0.901 | 1.000 | 63.261 | 0 |
| 3b. MiniLM + LogReg | 0.962 | 0.974 | 44.343 | 0 |
| 3c. MiniLM + MLP head | 0.966 | 0.947 | 59.152 | 0 |
| 4a. Gemini Flash Lite zero-shot | Chưa đo trên toàn TEST | 0.927 | 948.676 | 0.023 |
| 4c. Gemini Flash Lite few-shot | Chưa đo trên toàn TEST | 0.923 | 1860.367 | 0.048 |

**Nhận xét:** MiniLM + MLP head có F1 test thường cao nhất (0.966). Trên SUB 40, MiniLM + kNN đạt F1 = 1.000 nhưng p95 = 63.261 ms, vượt SLA 50 ms. BoW + MLP đạt F1 SUB = 0.974 với p95 chỉ 0.302 ms, là lựa chọn cân bằng tốt giữa độ chính xác và tốc độ trong phép đo này. F1 của LLM chỉ được đo trên SUB 40, không được xem là F1 toàn bộ TEST.

## 3. Dự đoán và thực tế

1. **Dự đoán về mô hình trên đường Pareto:** Tôi kỳ vọng MiniLM + LogReg thuộc nhóm tối ưu. **Thực tế:** MiniLM + LogReg bị BoW + MLP lấn át trên SUB 40: cùng F1 = 0.974 nhưng p95 lần lượt là 44.343 và 0.302 ms. **Giải thích:** Embedding tốn thời gian mã hóa văn bản, trong khi BoW xử lý nhanh và vẫn đủ hiệu quả với bộ dữ liệu này.
2. **Dự đoán về few-shot LLM:** Tôi kỳ vọng thêm ví dụ sẽ cải thiện độ chính xác so với zero-shot. **Thực tế:** F1 SUB của few-shot = 0.923, thấp hơn zero-shot = 0.927; p95 cũng tăng từ 948.676 lên 1860.367 ms. **Giải thích:** Ví dụ trong prompt không bảo đảm cải thiện trên mẫu nhỏ và làm tăng độ dài yêu cầu gửi đến API.
3. **Dự đoán về Stress Test:** Tôi dự đoán BoW giảm mạnh khi gặp spam tự nhiên hoặc né từ khóa, còn LLM có nguy cơ mắc lỗi ở prompt injection. **Thực tế:** Chưa có bảng tổng hợp vì lượt chạy Gemini bị ngắt giữa chừng. **Giải thích cần kiểm chứng:** Bộ case khó thay đổi cách biểu đạt so với dữ liệu huấn luyện, khiến mô hình dựa vào từ khóa dễ sai.

## 4. Phân tích 50 case khó

Bộ Stress Test có **50 tin**: 29 spam và 21 ham, chia thành bảy nhóm. Các bẫy gồm tin ham chứa từ khóa spam, spam diễn đạt tự nhiên, mạo danh, né từ khóa, prompt injection, tin giao dịch hợp lệ và tin nói về spam.

**Mô hình tụt nhiều nhất và nhóm khó nhất:** *Chưa xác định được — cần bảng F1 trên 50 case và kết quả theo nhóm.* Lượt chạy Gemini đầu tiên đạt 50/50 lượt gọi, nhưng quá trình bị ngắt khi Gemini tiếp theo mới đạt 3/50; chưa có bảng tổng hợp đáng tin cậy.

**Hai case cần đối chiếu với dự đoán thực tế:**
- Tin ham về vé miễn phí: “Txt me ur address and I'll send the free ticke...” — có từ “free”, dễ khiến BoW nhận nhầm là spam nếu phụ thuộc vào từ khóa. *Chưa xác nhận mô hình đã dự đoán sai.*
- Tin spam giả mạo cơ quan thuế: “HMRC: You are eligible for a tax refund of £32...” — tận dụng uy tín tổ chức và lời hứa hoàn tiền để đánh lừa người nhận. *Chưa xác nhận mô hình đã dự đoán sai.*

**Prompt injection:** Cần đối chiếu kết quả 4a (zero-shot) và 4d (prompt phòng thủ) trên nhóm `prompt_injection` và các nhóm khác trước khi kết luận phòng thủ có hiệu quả hay gây tác dụng phụ. Biện pháp bổ sung ngoài prompt: không cho nội dung SMS quyền thay đổi chỉ dẫn hệ thống, kết hợp bộ lọc độc lập hoặc kiểm tra quy tắc trước khi ra quyết định.

**Né từ khóa:** Có thể chuẩn hóa ký tự trước khi phân loại (ví dụ `C0ngr4tulat10ns` → `Congratulations`, ghép `F R E E` → `FREE`). Rủi ro: biến đổi nhầm từ hợp lệ, làm tăng false positive.

Bộ 50 case viết tay phù hợp để **phát hiện điểm yếu và kiểm thử độ bền**, không đủ đại diện để ước lượng chính xác hiệu năng trên lưu lượng SMS thực tế.

## 5. Thử thách cá nhân

- **SEED:** 9306.
- **Đề thử thách theo SEED:** *Cần chép nguyên văn từ notebook.*
- **Cách làm, kết quả, giải thích:** *Cần bổ sung sau khi chạy ô thử thách tương ứng.*

## 6. Phương án cho sản phẩm thật

Với yêu cầu **p95 < 50 ms/tin** và **10 triệu SMS/ngày**, tôi chọn **BoW + MLP** làm phương án nền. Mô hình đạt **F1 test thường = 0.942**, **F1 SUB 40 = 0.974**, **p95 = 0.302 ms/tin** và không phát sinh phí API cho mỗi tin. Trung bình 10 triệu tin/ngày tương đương khoảng **116 tin/giây**; throughput đo được của mô hình khoảng **21.988 tin/giây** trong môi trường thí nghiệm, cao hơn mức trung bình yêu cầu.

**Đánh đổi:** BoW + MLP có F1 test thường thấp hơn MiniLM + MLP head (0.942 so với 0.966), có thể kém bền vững khi gặp cách viết mới, né từ khóa hoặc spam theo ngữ cảnh. Cần bổ sung tiền xử lý, theo dõi false positive/false negative, đánh giá trên dữ liệu thực tế và kiểm thử tải giờ cao điểm trước khi triển khai. Các số liệu tốc độ trong notebook chưa thay thế được benchmark production.

## 7. AI log và điều đã học

| Nội dung hỏi AI | AI hỗ trợ | Tôi kiểm tra/chỉnh sửa |
| --- | --- | --- |
| Hoàn thiện hàm gọi API và kiểm thử | Gợi ý cấu trúc hàm, cách xử lý phản hồi và token usage | Chạy thử và sửa lỗi import, mock, định dạng kết quả theo yêu cầu bài lab |
| Chạy và đối chiếu các mô hình | Giải thích F1, latency, Pareto và SLA | Dùng số liệu notebook thực tế thay cho dự đoán ban đầu |
| Xử lý Stress Test bị ngắt | Giải thích `KeyboardInterrupt` khi chờ giữa các lời gọi Gemini | Tách phần đánh giá mô hình cục bộ khỏi lời gọi API để tránh chạy lại tốn thời gian |
| Viết phần nhận xét và báo cáo | Gợi ý cách diễn giải kết quả và rủi ro prompt injection | Giữ lại các kết luận có số liệu; đánh dấu phần chưa được kiểm chứng |

**Điều đã học:** F1 cao trên tập test cùng phân phối chưa bảo đảm mô hình chống chịu tốt với spam biến thể ngoài đời. Mô hình có ngữ nghĩa mạnh hơn chưa chắc phù hợp sản phẩm nếu độ trễ và chi phí cao. Cần phân biệt kết quả đo trên SUB 40 với toàn bộ TEST và không kết luận chắc chắn từ một tập mẫu quá nhỏ.
