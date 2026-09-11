# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.15 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  `class_id`: 468, `class_name`: "cab", `rank`: 1, `score`: 0.510915, `taxonomy_name`: "ImageNet-1K".
  (Các hạng tiếp theo trong top-5: rank 2: minibus - score 0.164284; rank 3: police_van - score 0.085848; rank 4: recreational_vehicle - score 0.05411; rank 5: streetcar - score 0.048193).
- Record này mô tả toàn ảnh như thế nào?
  Mô tả ở cấp độ toàn cục (global image-level) — gán nhãn duy nhất cho toàn bộ bức ảnh (nhận định ảnh thuộc lớp "cab" với độ tự tin ~51.09%), không chỉ ra vị trí, ranh giới hay tọa độ của vật thể nằm ở đâu trong khung hình.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Do nhóm nghiên cứu/tác giả của tập dữ liệu huấn luyện (ImageNet-1K) định nghĩa từ trước với 1.000 lớp cố định. Checkpoint chỉ tính toán và xếp hạng xác suất cho các lớp trong danh sách cố định đó, không thể tự suy nghĩ ra lớp mới nằm ngoài taxonomy.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id` giúp hệ thống máy tính lập chỉ mục (index) và tính toán loss/metrics chuẩn xác, tiết kiệm dung lượng.
  - `class_name` giúp con người (kỹ sư, annotator, reviewer) đọc hiểu trực quan.
  - `taxonomy_name` đóng vai trò không gian tên (namespace) quan trọng giúp tránh nhầm lẫn giữa các chuẩn gán nhãn khác nhau (ví dụ: `class_id = 0` ở ImageNet-1K là loài cá "tench", trong khi `class_id = 0` ở COCO là "person").
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Cần quy định rõ quy tắc ưu tiên (precedence rule): ưu tiên theo chủ thể chiếm diện tích lớn nhất trong khung hình, chủ thể nằm ở vị trí trung tâm, hoặc theo mục tiêu nghiệp vụ của dự án (ví dụ: bài toán giao thông ưu tiên phương tiện hơn cảnh quan phông nền); hoặc cần chuyển đổi bài toán sang Multi-label Classification (cho phép gán nhiều nhãn đồng thời cho ảnh).
- Vì sao model score không phải ground truth?
  Model score chỉ là điểm số tự tin toán học (confidence probability) sau hàm softmax dựa trên phân phối trọng số mà mô hình đã học, không phản ánh sự thật khách quan của thế giới thực. Mô hình hoàn toàn có thể đạt score cao (>0.9) nhưng vẫn phán đoán sai (overconfidence), hoặc ngược lại score thấp do phân vân giữa hai lớp tương đồng. Ground truth phải do con người kiểm chứng và xác nhận độc lập dựa trên guideline gán nhãn nghiêm ngặt.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  `class_name`: "oven", `score`: 0.686791, `bbox_xyxy`: [0.11, 187.54, 195.96, 292.81], `bbox_width`: 195.85, `bbox_height`: 105.26.
  (Hoặc record: `class_name`: "person", `score`: 0.912625, `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92], `bbox_width`: 113.58, `bbox_height`: 279.68).
- Diễn giải vị trí box bằng lời:
  Hộp bao quanh chiếc lò nướng (oven) nằm trên bệ bếp phía bên trái khung hình; góc trên bên trái box nằm sát mép trái ảnh tại tọa độ pixel (0.11, 187.54) và góc dưới bên phải tại tọa độ (195.96, 292.81); chiều rộng hộp là 195.85 px và chiều cao là 105.26 px.
- So sánh số prediction ở hai threshold:
  Dựa trên kết quả chạy thực nghiệm trong notebook với sample `kitchen`:
  - Ở threshold thấp (conf = 0.20): Phát hiện 17 vật thể (gồm person, bowl, oven, cup, spoon, potted plant, dining table, bottle). Số lượng box xuất hiện nhiều hơn, bắt được cả các vật thể nhỏ hoặc bị che khuất một phần (recall cao) nhưng có nhiều box nhiễu/sai (false positives).
  - Ở threshold mặc định (conf = 0.35): Giữ lại 11 vật thể (2 person, 5 bowl, 2 oven, 2 cup). Cân bằng giữa độ nhạy và độ chính xác.
  - Ở threshold cao (conf = 0.60): Số lượng box giảm rõ rệt xuống còn 6 vật thể (2 person, 2 bowl, 2 oven). Chỉ giữ lại các vật thể có độ tin cậy cao, loại bỏ được nhiều dự đoán rác nhưng dễ bỏ sót vật thể thực tế (missed detections / false negatives).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  Threshold thấp tăng độ bao phủ (coverage / recall) của các đối tượng trong ảnh nhưng tạo áp lực lớn cho reviewer vì phải lọc và xóa bỏ nhiều nhãn rác. Threshold cao giảm khối lượng việc cho reviewer (chỉ cần duyệt các dự đoán chắc chắn), nhưng làm tăng nguy cơ bỏ sót vật thể thật cần gán nhãn, buộc người gán nhãn phải vẽ bù thủ công.
- Đề xuất một quy tắc box chặt:
  Bounding box phải ôm sát đường viền ngoài cùng nhìn thấy được của đối tượng (dung sai thừa không quá 2–3 pixel), không được cắt lẹm vào thân hay các chi tiết của vật thể và không được chứa khoảng trống nền (background) không cần thiết.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  Guideline phải nêu rõ: Tỷ lệ che khuất tối đa được phép gán nhãn (ví dụ: che trên 80% thì bỏ qua); chỉ vẽ box bao quanh phần thực tế nhìn thấy (visible box) hay vẽ ước lượng toàn thể cả phần bị che (amodal box); và quy tắc khi vật thể chạm/bị cắt bởi mép ảnh. Các trường hợp mơ hồ không thể xác định lớp buộc annotator phải escalate (báo lên cấp trên/QC) để phân xử.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  `instance_id`: "kitchen-002", `class_name`: "bowl", `score`: 0.735743, số điểm đa giác: 67 điểm, một phần tọa độ polygon_xy: `[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0], ...]`, `bbox_xyxy`: [31.44, 343.77, 100.89, 385.55].
  (Hoặc record: `instance_id`: "kitchen-007", `class_name`: "oven", `score`: 0.489667, số điểm đa giác: 213 điểm, một phần tọa độ polygon_xy: `[[1.0, 189.0], [1.0, 218.0], [1.0, 214.0], [2.0, 213.0], [2.0, 201.0], ...]`, `bbox_xyxy`: [0.25, 188.95, 194.35, 294.62]).
- Polygon bổ sung chi tiết gì so với box?
  Polygon mô tả chính xác đường biên hình học ở cấp độ pixel (pixel-level contour) theo đúng hình dáng thực tế của đối tượng, loại bỏ hoàn toàn phần pixel nền xung quanh mà bounding box hình chữ nhật bắt buộc phải chứa.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  `instance_id` dùng để định danh và phân tách riêng biệt từng cá thể đối tượng trong cùng một lớp trong một bức ảnh (instance segmentation). Nó không phải là `class_id` (mã định danh lớp chung) và không phải là `tracking_id` (mã bám vết vật thể qua chuỗi video).
- Đề xuất một quy tắc biên mask:
  Các điểm nút đa giác (vertices) phải bám khít đường biên vật thể với độ lệch không quá 1–2 pixel; đường cong cần đủ số điểm để không bị gãy góc thô; tuyệt đối không vẽ đè lên bóng đổ (shadow) của vật thể.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Cần quy định: Cách phân tách ranh giới khi hai vật thể cùng màu dính sát vào nhau (kissing/touching objects); vùng bị mờ do chuyển động (motion blur) lấy đường biên tới đâu; và khi bị vật thể khác cắt ngang ở giữa thì gán thành 1 mask dạng multi-polygon hay tách thành 2 instance. Khi ranh giới không thể phân biệt bằng mắt, phải escalate lên cấp quản lý / QC.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn/Class ID toàn ảnh (Image-level label) | Ảnh có nhiều đối tượng cạnh tranh nhau (taxi, minibus, cảnh sát...); mô hình gán nhãn "cab" chỉ đại diện 1 phần ảnh, bỏ qua toàn cảnh | Quan sát toàn diện ảnh, đối chiếu quy tắc ưu tiên chủ thể trong guideline để chọn 1 nhãn chuẩn nhất | Kiểm tra nhãn được chọn có đúng quy tắc ưu tiên của dự án hay không; rà soát các ảnh mơ hồ (edge cases) |
| Phát hiện vật thể | Bounding box 2D [x_min, y_min, x_max, y_max] + Class ID | Box bị lỏng thừa nhiều nền; sót các vật thể nhỏ/xa (thìa, lọ hoa); vẽ trùng box lên cùng vật thể | Xác định từng cá thể, vẽ hộp ôm khít viền ngoài cùng nhìn thấy được, gắn đúng lớp | Kiểm tra độ ôm khít (tightness), tỷ lệ bỏ sót/vẽ thừa (IoU), xử lý đúng các vật thể chạm mép hoặc bị che khuất |
| Instance segmentation | Đa giác khép kín (Polygon [[x,y],...]) + Instance ID + Class ID | Ranh giới mask cắt lẹm vào vật thể; gộp nhầm 2 vật thể đứng sát nhau thành một cá thể; vẽ lẫn bóng đổ | Chấm các điểm đa giác viền khít từng đối tượng riêng biệt; gán đúng instance_id và class_name | Soi độ chính xác đường biên mask ở mức pixel; kiểm tra các điểm tiếp xúc giữa các vật thể và xử lý phần bị che |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Tuyệt đối không tải lên (upload) ảnh cá nhân, chân dung, biển số xe, thông tin nhận dạng cá nhân (PII), dữ liệu khách hàng hoặc dữ liệu nội bộ chưa được cấp phép lên notebook hoặc kho GitHub công khai. Chỉ sử dụng tập ảnh công khai có bản quyền được cung cấp cho bài lab.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Giảng viên hướng dẫn / Lab Coach phụ trách lớp học.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
