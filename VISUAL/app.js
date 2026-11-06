// VISUAL/app.js — Logic tương tác và nạp chỉ số động cho Local Dashboard (Practice 3)

// 1. Lightbox Modal cho xem ảnh phóng to
function openModal(src) {
  const modal = document.getElementById('img-modal');
  const target = document.getElementById('modal-target');
  if (modal && target) {
    target.src = src;
    modal.classList.add('active');
  }
}

function closeModal() {
  const modal = document.getElementById('img-modal');
  if (modal) {
    modal.classList.remove('active');
  }
}

// Phím ESC để đóng lightbox
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeModal();
  }
});

// 2. Tự động tải số liệu động từ outputs/reports/final_metrics.json (P1-11)
async function loadDynamicMetrics() {
  try {
    const response = await fetch('../outputs/reports/final_metrics.json');
    if (!response.ok) {
      console.warn('Không thể đọc file final_metrics.json, sử dụng số liệu chuẩn hóa mặc định.');
      return;
    }
    const data = await response.json();
    console.log('✅ Đã nạp thành công metrics động:', data);

    const testAcc = data.test_accuracy ?? (data.test_accuracy_ratio ? data.test_accuracy_ratio * 100 : 74.5);
    const majority = data.majority_baseline ?? 52.0;
    const untrained = data.untrained_head_val_baseline ?? 44.0;
    const correct = data.correct_predictions ?? 149;
    const total = data.total_samples ?? 200;
    const errors = data.error_count ?? (total - correct);
    const ci = data.wilson_95_ci ?? [68.04, 80.04];
    const f1 = data.test_macro_f1 ?? (data.test_macro_f1_ratio ? data.test_macro_f1_ratio * 100 : 74.27);
    const precision = data.test_precision ?? (data.test_precision_ratio ? data.test_precision_ratio * 100 : 74.76);
    const recall = data.test_recall ?? (data.test_recall_ratio ? data.test_recall_ratio * 100 : 74.24);
    const loss = data.test_loss ?? 0.5565;

    // Cập nhật DOM nếu các phần tử tồn tại
    const elScore = document.getElementById('dyn-score-number');
    if (elScore) elScore.textContent = `${testAcc.toFixed(2)}%`;

    const elSubtext = document.getElementById('dyn-score-subtext');
    if (elSubtext) elSubtext.textContent = `${correct} / ${total} bài đánh giá phim được phân loại chính xác nhãn POSITIVE / NEGATIVE`;

    const elGainMaj = document.getElementById('dyn-gain-majority');
    if (elGainMaj) elGainMaj.textContent = `▲ +${(testAcc - majority).toFixed(2)} điểm %`;

    const elGainVal = document.getElementById('dyn-gain-val');
    if (elGainVal) {
      const valAcc = data.finetuned_val_accuracy ?? 76.5;
      elGainVal.textContent = `▲ +${(valAcc - untrained).toFixed(2)} điểm %`;
    }

    const elGainUntrained = document.getElementById('dyn-gain-untrained');
    if (elGainUntrained) elGainUntrained.textContent = `+${(testAcc - untrained).toFixed(2)} điểm %`;

    const elCi = document.getElementById('dyn-ci-text');
    if (elCi) elCi.textContent = `Khoảng tin cậy Wilson: [${ci[0].toFixed(2)}%, ${ci[1].toFixed(2)}%]`;

    const elTblAcc = document.getElementById('dyn-tbl-acc');
    if (elTblAcc) elTblAcc.textContent = `${testAcc.toFixed(2)}%`;

    const elTblF1 = document.getElementById('dyn-tbl-f1');
    if (elTblF1) elTblF1.textContent = `${f1.toFixed(2)}%`;

    const elTblPrec = document.getElementById('dyn-tbl-prec');
    if (elTblPrec) elTblPrec.textContent = `${precision.toFixed(2)}%`;

    const elTblRec = document.getElementById('dyn-tbl-rec');
    if (elTblRec) elTblRec.textContent = `${recall.toFixed(2)}%`;

    const elTblLoss = document.getElementById('dyn-tbl-loss');
    if (elTblLoss) elTblLoss.textContent = `${loss.toFixed(4)}`;

    const elErrorCountHeader = document.getElementById('dyn-error-count-header');
    if (elErrorCountHeader) elErrorCountHeader.textContent = `6. Nghiên Cứu Điển Hình ${errors} Ca Sai Số (Error Analysis)`;

    const elLiveBadge = document.querySelector('.live-badge span:last-child');
    if (elLiveBadge) elLiveBadge.textContent = 'Dữ Liệu Động (Đã Đồng Bộ)';
  } catch (err) {
    console.error('Lỗi khi tải final_metrics.json:', err);
  }
}

// Khởi chạy nạp metrics khi trang sẵn sàng
document.addEventListener('DOMContentLoaded', () => {
  loadDynamicMetrics();
});
